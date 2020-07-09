#include "CylinderMap.hpp"
extern "C" {
    void init(int width, int height) {
        printf("CylinderMap init with width and height(%d; %d)\n", width, height);
        CylinderMap::init(width, height);
    }
    void PointCloudToCylinderMap(float* point, float *feature, int point_size, float *depth_map, float *result, int feature_dim, float *ids, float *landmarks, int n_landmark, float *landmark_ids) { //result: H*W*k; point_dim: 3; ids:point_size*2
        CylinderMap::getUVMap(point, feature, point_size, depth_map, result, feature_dim, ids, landmarks, n_landmark, landmark_ids);
    }
}

