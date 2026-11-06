#!/usr/bin/env python

"""Unit tests for second-pass fixes in extract, tessera, and classify."""

import importlib.machinery
import sys
import tempfile
import types
import unittest
from unittest.mock import MagicMock, patch

import geopandas as gpd
import numpy as np
import rasterio
from rasterio.transform import from_bounds, from_origin
from shapely.geometry import Point, box
import torch


def _ensure_optional_import_stubs():
    """Provide minimal module stubs when optional dependencies are absent."""
    for mod_name in [
        "matplotlib",
        "matplotlib.pyplot",
        "leafmap",
        "rioxarray",
        "xarray",
        "segment_geospatial",
        "samgeo",
    ]:
        try:
            __import__(mod_name)
        except ImportError:
            stub = MagicMock()
            stub.__spec__ = importlib.machinery.ModuleSpec(mod_name, None)
            sys.modules.setdefault(mod_name, stub)

    try:
        import torchgeo.datasets  # noqa: F401
    except ImportError:
        spec_fn = importlib.machinery.ModuleSpec
        tg = types.ModuleType("torchgeo")
        tg.__spec__ = spec_fn("torchgeo", None)
        tg_datasets = types.ModuleType("torchgeo.datasets")
        tg_datasets.__spec__ = spec_fn("torchgeo.datasets", None)

        class _NonGeoDataset:
            pass

        tg_datasets.NonGeoDataset = _NonGeoDataset
        tg.datasets = tg_datasets
        sys.modules.setdefault("torchgeo", tg)
        sys.modules.setdefault("torchgeo.datasets", tg_datasets)


_ensure_optional_import_stubs()

from geoai import classify  # noqa: E402
from geoai.embeddings import _merge_tiles  # noqa: E402
from geoai.extract import ObjectDetector  # noqa: E402
from geoai.tessera import tessera_sample_points  # noqa: E402


class TestExtractSecondPass(unittest.TestCase):
    """Tests for STRtree NMS, edge filtering, and bbox contour extraction."""

    def _make_detector_stub(self):
        detector = ObjectDetector.__new__(ObjectDetector)
        detector.nms_iou_threshold = 0.5
        detector.simplify_tolerance = 0.0
        detector.mask_threshold = 0.5
        detector.min_object_area = 10
        detector.max_object_area = None
        return detector

    def test_filter_overlapping_polygons_strtree_equivalence(self):
        """STRtree NMS matches naive pairwise IoU filtering."""
        det = self._make_detector_stub()
        polys = [
            box(0, 0, 10, 10),  # idx 0: high confidence
            box(1, 1, 10, 10),  # idx 1: overlaps idx 0 (IoU = 0.81) -> drop
            box(8, 8, 18, 18),  # idx 2: slight overlap (IoU = 0.02) -> keep
            box(20, 20, 30, 30),  # idx 3: disjoint -> keep
            box(20.5, 20.5, 29.5, 29.5),  # idx 4: overlaps idx 3 -> drop
        ]
        confidences = [0.95, 0.80, 0.75, 0.90, 0.60]
        gdf = gpd.GeoDataFrame(
            {"geometry": polys, "confidence": confidences},
            crs="EPSG:32610",
        )

        filtered = det.filter_overlapping_polygons(gdf, nms_iou_threshold=0.5)

        self.assertEqual(list(filtered.index), [0, 3, 2])
        self.assertEqual(len(filtered), 3)

    def test_filter_edge_objects_vectorized(self):
        """Vectorized filter_edge_objects keeps >=50% interior polygons."""
        det = self._make_detector_stub()
        with tempfile.TemporaryDirectory() as tmpdir:
            raster_path = f"{tmpdir}/test_raster.tif"
            transform = from_origin(0.0, 100.0, 1.0, 1.0)
            with rasterio.open(
                raster_path,
                "w",
                driver="GTiff",
                height=100,
                width=100,
                count=1,
                dtype="uint8",
                crs="EPSG:32610",
                transform=transform,
            ) as dst:
                dst.write(np.zeros((1, 100, 100), dtype=np.uint8))

            # inner_box with edge_buffer=10 is box(10, 10, 90, 90)
            polys = [
                box(20, 20, 40, 40),  # 100% inside inner_box -> keep
                box(5, 20, 20, 30),  # 66.7% inside inner_box -> keep
                box(2, 20, 15, 30),  # 38.5% inside inner_box -> drop
                box(0, 0, 5, 5),  # 0% inside inner_box -> drop
            ]
            gdf = gpd.GeoDataFrame({"geometry": polys}, crs="EPSG:32610")

            res = det.filter_edge_objects(gdf, raster_path, edge_buffer=10)
            self.assertEqual(list(res.index), [0, 1])

    def test_masks_to_vector_bbox_sliced_contours(self):
        """masks_to_vector preserves global offsets when slicing bboxes."""
        det = self._make_detector_stub()
        with tempfile.TemporaryDirectory() as tmpdir:
            mask_path = f"{tmpdir}/mask.tif"
            mask = np.zeros((200, 200), dtype=np.uint8)
            mask[20:40, 30:60] = 255
            mask[100:150, 120:160] = 255

            transform = from_origin(1000.0, 2000.0, 1.0, 1.0)
            with rasterio.open(
                mask_path,
                "w",
                driver="GTiff",
                height=200,
                width=200,
                count=1,
                dtype="uint8",
                crs="EPSG:32610",
                transform=transform,
            ) as dst:
                dst.write(mask, 1)

            gdf = det.masks_to_vector(
                mask_path,
                regularize=False,
                min_object_area=10,
                simplify_tolerance=0.0,
            )

            self.assertIsNotNone(gdf)
            self.assertEqual(len(gdf), 2)
            raw_bounds = []
            for g in gdf.geometry:
                raw_bounds.append(tuple(round(v, 1) for v in g.bounds))
            bounds_sorted = sorted(raw_bounds)
            self.assertEqual(
                bounds_sorted[0],
                (1030.0, 1961.0, 1059.0, 1980.0),
            )
            self.assertEqual(
                bounds_sorted[1],
                (1120.0, 1851.0, 1159.0, 1900.0),
            )


class TestTesseraSamplePointsSecondPass(unittest.TestCase):
    """Tests for tessera_sample_points tile rounding and pixel flooring."""

    def test_upper_half_tile_lookup_and_negative_pixel_floor(self):
        """Point at lon=0.16 in [0.10, 0.20) maps to tile 0.15, not 0.25."""
        tile_015_emb = np.full((10, 10, 128), 15.0, dtype=np.float32)
        tile_025_emb = np.full((10, 10, 128), 25.0, dtype=np.float32)
        transform_015 = from_bounds(0.10, 0.10, 0.20, 0.20, 10, 10)
        transform_025 = from_bounds(0.20, 0.20, 0.30, 0.30, 10, 10)

        fake_gt = MagicMock()
        fake_gt.registry.load_blocks_for_region.return_value = ["b1", "b2"]
        fake_gt.fetch_embeddings.return_value = [
            (2024, 0.15, 0.15, tile_015_emb, "EPSG:4326", transform_015),
            (2024, 0.25, 0.25, tile_025_emb, "EPSG:4326", transform_025),
        ]
        fake_gt_module = MagicMock()
        fake_gt_module.GeoTessera.return_value = fake_gt

        points_gdf = gpd.GeoDataFrame(
            geometry=[Point(0.16, 0.16), Point(0.04, 0.16)],
            crs="EPSG:4326",
        )
        with (
            patch.dict(sys.modules, {"geotessera": fake_gt_module}),
            patch(
                "geoai.tessera._check_geotessera",
                return_value=fake_gt_module,
            ),
        ):
            result = tessera_sample_points(points_gdf, year=2024)

        self.assertAlmostEqual(result.loc[0, "tessera_0"], 15.0)
        self.assertTrue(np.isnan(result.loc[1, "tessera_0"]))


class TestClassifySecondPass(unittest.TestCase):
    """Tests for AlbumentationsWrapper and CoarseDropout 2.x compatibility."""

    def _run_train_classifier_with_mock_env(self, coarse_dropout_v2=True):
        captured = {
            "raster_init_transforms": [],
            "intersection_transforms": [],
        }

        fake_a = types.ModuleType("albumentations")
        for name in [
            "HorizontalFlip",
            "VerticalFlip",
            "RandomRotate90",
            "ShiftScaleRotate",
            "RandomBrightnessContrast",
            "GaussianBlur",
            "GaussNoise",
        ]:
            setattr(fake_a, name, MagicMock(return_value=name))

        if coarse_dropout_v2:

            class FakeCoarseDropoutV2:
                def __init__(
                    self,
                    num_holes_range=(1, 1),
                    hole_height_range=(8, 8),
                    hole_width_range=(8, 8),
                    p=0.5,
                ):
                    captured["coarse_dropout_kwargs"] = {
                        "num_holes_range": num_holes_range,
                        "hole_height_range": hole_height_range,
                        "hole_width_range": hole_width_range,
                        "p": p,
                    }

            fake_a.CoarseDropout = FakeCoarseDropoutV2
        else:

            class FakeCoarseDropoutV1:
                def __init__(
                    self,
                    max_holes=8,
                    max_height=8,
                    max_width=8,
                    p=0.5,
                ):
                    captured["coarse_dropout_kwargs"] = {
                        "max_holes": max_holes,
                        "max_height": max_height,
                        "max_width": max_width,
                        "p": p,
                    }

            fake_a.CoarseDropout = FakeCoarseDropoutV1

        def fake_compose(transforms_list):
            def _apply(image, mask):
                captured["transform_mask_dtype"] = mask.dtype
                return {
                    "image": image[:, ::-1, :].copy(),
                    "mask": mask[:, ::-1].copy(),
                }

            return _apply

        fake_a.Compose = fake_compose

        fake_pl = types.ModuleType("lightning.pytorch")
        fake_pl_callbacks = types.ModuleType("lightning.pytorch.callbacks")
        fake_pl_loggers = types.ModuleType("lightning.pytorch.loggers")
        fake_pl_callbacks.ModelCheckpoint = MagicMock()
        fake_pl_loggers.CSVLogger = MagicMock()

        class FakeTrainer:
            def __init__(self, **kwargs):
                pass

            def fit(self, model, datamodule, ckpt_path=None):
                datamodule.setup("fit")

            def test(self, model, datamodule):
                pass

        fake_pl.Trainer = FakeTrainer
        fake_lightning = types.ModuleType("lightning")
        fake_lightning.pytorch = fake_pl

        fake_tg_dm = types.ModuleType("torchgeo.datamodules")

        class FakeGeoDataModule:
            def __init__(
                self,
                dataset_class,
                batch_size=1,
                patch_size=64,
                length=10,
                num_workers=0,
                **kwargs,
            ):
                self.dataset_class = dataset_class
                self.batch_size = batch_size
                self.patch_size = patch_size
                self.length = length
                self.num_workers = num_workers
                self.kwargs = kwargs

        fake_tg_dm.GeoDataModule = FakeGeoDataModule

        fake_tg_ds = types.ModuleType("torchgeo.datasets")

        class FakeRasterDataset:
            def __init__(self, paths=None, transforms=None, **kwargs):
                self.paths = paths
                self.transforms = transforms
                captured["raster_init_transforms"].append(transforms)

        class FakeIntersectionDataset:
            def __init__(self, dataset1, dataset2, transforms=None):
                self.dataset1 = dataset1
                self.dataset2 = dataset2
                self.transforms = transforms
                captured["intersection_transforms"].append(transforms)

        fake_tg_ds.RasterDataset = FakeRasterDataset
        fake_tg_ds.IntersectionDataset = FakeIntersectionDataset
        fake_tg_ds.stack_samples = MagicMock()

        fake_tg_splits = types.ModuleType("torchgeo.datasets.splits")

        def _fake_split(dataset, split, generator):
            return dataset, dataset, dataset

        fake_tg_splits.random_bbox_assignment = _fake_split

        fake_tg_samplers = types.ModuleType("torchgeo.samplers")
        fake_tg_samplers.GridGeoSampler = MagicMock()
        fake_tg_samplers.RandomBatchGeoSampler = MagicMock()
        fake_tg_samplers.RandomGeoSampler = MagicMock()

        fake_tg_trainers = types.ModuleType("torchgeo.trainers")
        mock_task_cls = MagicMock(return_value="mock_task")
        fake_tg_trainers.SemanticSegmentationTask = mock_task_cls

        with tempfile.TemporaryDirectory() as tmpdir:
            with patch.dict(
                sys.modules,
                {
                    "albumentations": fake_a,
                    "lightning": fake_lightning,
                    "lightning.pytorch": fake_pl,
                    "lightning.pytorch.callbacks": fake_pl_callbacks,
                    "lightning.pytorch.loggers": fake_pl_loggers,
                    "torchgeo.datamodules": fake_tg_dm,
                    "torchgeo.datasets": fake_tg_ds,
                    "torchgeo.datasets.splits": fake_tg_splits,
                    "torchgeo.samplers": fake_tg_samplers,
                    "torchgeo.trainers": fake_tg_trainers,
                },
            ):
                classify.train_classifier(
                    image_root=tmpdir,
                    label_root=tmpdir,
                    output_dir=tmpdir,
                    use_augmentation=True,
                    callbacks=[],
                )

        return captured

    def test_coarse_dropout_albumentations_v2_and_joint_wrapper(self):
        """Albumentations 2.x CoarseDropout and wrapper preserve mask shape."""
        v2_flag = True
        captured = self._run_train_classifier_with_mock_env(v2_flag)

        self.assertEqual(
            captured["coarse_dropout_kwargs"],
            {
                "num_holes_range": (1, 8),
                "hole_height_range": (8, 32),
                "hole_width_range": (8, 32),
                "p": 0.3,
            },
        )
        self.assertEqual(captured["raster_init_transforms"], [None, None])

        wrapper = captured["intersection_transforms"][-1]
        self.assertIsNotNone(wrapper)

        img = torch.arange(12, dtype=torch.float32).reshape(3, 2, 2)
        mask_2d = torch.tensor([[1, 2], [3, 4]], dtype=torch.int64)
        out_2d = wrapper({"image": img, "mask": mask_2d})
        self.assertEqual(captured["transform_mask_dtype"], np.int32)
        self.assertEqual(out_2d["mask"].shape, (2, 2))
        self.assertEqual(out_2d["mask"].dtype, torch.int64)
        expected_2d = torch.tensor([[2, 1], [4, 3]], dtype=torch.int64)
        self.assertTrue(torch.equal(out_2d["mask"], expected_2d))

        mask_3d = mask_2d.unsqueeze(0)
        out_3d = wrapper({"image": img, "mask": mask_3d})
        self.assertEqual(out_3d["mask"].shape, (1, 2, 2))
        self.assertEqual(out_3d["mask"].dtype, torch.int64)

    def test_coarse_dropout_albumentations_v1_fallback(self):
        """Albumentations 1.x CoarseDropout uses max_holes/height/width."""
        v2_flag = False
        captured = self._run_train_classifier_with_mock_env(v2_flag)
        self.assertEqual(
            captured["coarse_dropout_kwargs"],
            {
                "max_holes": 8,
                "max_height": 32,
                "max_width": 32,
                "p": 0.3,
            },
        )


class TestEmbeddingsMergeTilesSecondPass(unittest.TestCase):
    """Tests for _merge_tiles preserving nodata=-128."""

    def test_merge_tiles_preserves_nodata_minus_128(self):
        """Overlapping int8 tiles treat -128 as nodata during merge."""
        t1_data = np.array([[[50, -128], [50, -128]]], dtype=np.int8)
        t2_data = np.array([[[80, 80], [80, 80]]], dtype=np.int8)

        tile_arrays = [
            {
                "data": t1_data,
                "bounds": (0.0, 0.0, 2.0, 2.0),
                "crs": "EPSG:32610",
            },
            {
                "data": t2_data,
                "bounds": (1.0, 0.0, 3.0, 2.0),
                "crs": "EPSG:32610",
            },
        ]

        mosaic, bounds = _merge_tiles(tile_arrays)
        self.assertEqual(bounds, (0.0, 0.0, 3.0, 2.0))
        self.assertEqual(mosaic.shape, (1, 2, 3))
        expected = np.array([[[50, 80, 80], [50, 80, 80]]], dtype=np.int8)
        np.testing.assert_array_equal(mosaic, expected)


if __name__ == "__main__":
    unittest.main()
