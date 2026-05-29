"""Configuration enums for clustering algorithms."""

import enum


class ClusterNumberDetection(enum.Enum):
    DISTORTION = 1
    SILHOUETTE = 2
    CALINSKI_HARABASZ = 3
