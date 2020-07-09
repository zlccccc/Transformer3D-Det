#ifndef _CylinderMap_HPP
#define _CylinderMap_HPP

#include <cstdio>
#include <algorithm>
#include <cmath>
#include <cstring>
#define M_PI    3.14159265358979323846

namespace CylinderMap {
    int width, height; //const value; should init before ask
    float Ymin, Ystep, Astep, Rmax, Zbias;

    void init(int Adiv, unsigned Ydiv) {
        width = Adiv;
        height = Ydiv;
        float Amax = M_PI;
        Rmax = 3.6;
        Ystep = 3.2 / static_cast<float>(Ydiv);
        Astep = Amax * 2 / static_cast<float>(Adiv - 1);
        Zbias = 0.2;
    }

    int Point2Index(const float v[3], float id[2]) {
        id[0] = v[1] / Ystep + static_cast<float>(height) / 2 - 0.5;// 0 to height-1
        id[1] = atan2(v[0], v[2]+Zbias) / Astep + static_cast<float>(width) / 2 - 0.5;
        int y = static_cast<int>(id[0]+0.5), a = static_cast<int>(id[1]+0.5);
        std::swap(id[0], id[1]); //map[x,y]-point[v]
        if (y < 0 || a < 0 || y >= height || a >= width)
            return -1;
        return a + y * width;
    }

    bool getUVMap(const float *clds, const float *feature_cld, int n, float *depth_map, float *feature_map, const int dim, float *ids, const float *landmarks, int n_landmark, float *landmark_ids, int spread=3) { //for every point and put it in
        bool ret=0; //spread: spread_point_number
        int *weight = (int*)malloc(width * height * sizeof(int));
        int *Q = nullptr, front=0, end=0, maxend=std::min(width * height, n*spread);
        if (spread) Q = (int*)malloc(maxend * sizeof(int));
        std::memset(weight, 0, width * height * sizeof(int));
        for (int i = 0; i < n; i++, clds += 3, feature_cld += dim, ids+=2) {
            float r = std::sqrt(clds[0] * clds[0] + clds[2] * clds[2]);
            if (r > Rmax) {
                printf("warning: point %d out of range(too far from middle); (%f, %f, %f) dist=%f\n", i, clds[0], clds[1], clds[2], std::sqrt(clds[0] * clds[0] + clds[2] * clds[2]));
                ret = 1;
                continue;
            }
            int u = Point2Index(clds, ids);
            if (u < 0) {
                printf("warning: cannot find point: index out of range %d; (%f, %f, %f); id=(%f, %f), uxy(realposition)=(%d, %d)\n", i, clds[0], clds[1], clds[2], ids[0], ids[1], u/width, u%width);
                ret = 1;
                continue;
            }
            depth_map[u] = std::max(depth_map[u], r);
            weight[u] ++;
            for (int k = 0; k < dim; k++)
                feature_map[u * dim + k] += feature_cld[k];
        }
        for (int i = 0; i < n_landmark; i++, landmarks += 3, landmark_ids += 2) {
            int u = Point2Index(landmarks, landmark_ids);
            if (u < 0) {
                printf("warning: cannot find landmark point %d: index out of range; (%f, %f, %f); id=(%f, %f)\n", i, clds[0], clds[1], clds[2], landmark_ids[0], landmark_ids[1]);
                continue;
            }
        }
            
        for (int i = 0; i < width * height; i++)
            if (weight[i]) {
                for (int k = 0; k < dim; k++)
                    feature_map[i * dim + k] /= weight[i];
                if (end<maxend) Q[end++] = i;
                else if (spread) puts("WRONG!!! SPREAD WRONG");
            }
        if (spread) {//bfs it; same as nearest
            while (front<end) {
                int id=Q[front++], x=id/width, y=id-x*width;
                if (y!=0&&!weight[id-1]) {//left
                    int p=id-1; weight[p]=1;
                    depth_map[p]=depth_map[id];
                    for (int k = 0; k < dim; k++)
                        feature_map[p * dim + k] = feature_map[id*dim+k];
                    if (end<maxend) Q[end++]=p;
                }
                if (y!=width-1&&!weight[id+1]) {//left
                    int p=id+1; weight[p]=1;
                    depth_map[p]=depth_map[id];
                    for (int k = 0; k < dim; k++)
                        feature_map[p * dim + k] = feature_map[id*dim+k];
                    if (end<maxend) Q[end++]=p;
                }
                if (x!=0&&!weight[id-width]) {//left
                    int p=id-width; weight[p]=1;
                    depth_map[p]=depth_map[id];
                    for (int k = 0; k < dim; k++)
                        feature_map[p * dim + k] = feature_map[id*dim+k];
                    if (end<maxend) Q[end++]=p;
                }
                if (x!=height-1&&!weight[id+width]) {//left
                    int p=id+width; weight[p]=1;
                    depth_map[p]=depth_map[id];
                    for (int k = 0; k < dim; k++)
                        feature_map[p * dim + k] = feature_map[id*dim+k];
                    if (end<maxend) Q[end++]=p;
                }
            }
            free(Q);
        }
        free(weight);
        return ret;
    }
    // remove uvmap_to_point_cloud(with_face)_codes
}
#endif //_CylinderMap_HPP z

