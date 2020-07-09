#ifndef _KD_TREE_HPP_
#define _KD_TREE_HPP_
#include "cuda_array.hpp"
#include <algorithm>
#include <math.h>
#include <stdio.h>
namespace KDTrees {
const int DIM = 3;
struct Point {
    float A[DIM], max[DIM], min[DIM];
    int l, r;
    void init()
    {
        for (int i = 0; i < DIM; i++)
            min[i] = max[i] = A[i];
        l = r = -1;
    }
};
struct PointQuery {
    float A[DIM];
};
Point* T = nullptr;
int Cur, *id;
bool cmp(int x, int y)
{
    return T[x].A[Cur] < T[y].A[Cur];
}
void build_nodes(int& x, int l, int r, int cur, const int DIM = 3)
{ //should have id
    x = -1;
    //printf("building %d %d %d\n", l, r, cur);
    if (l >= r)
        return;
    int m = (l + r - 1) / 2;
    Cur = cur;
    std::nth_element(id + l, id + m, id + r, cmp);
    x = id[m];
    build_nodes(T[x].l, l, m, (cur + 1) % DIM, DIM); //l to m-1
    build_nodes(T[x].r, m + 1, r, (cur + 1) % DIM, DIM); // m+1 to r
    l = T[x].l, r = T[x].r;
    for (int i = 0; i < DIM; i++) {
        if (l != -1) {
            T[x].max[i] = max(T[x].max[i], T[l].max[i]);
            T[x].min[i] = min(T[x].min[i], T[l].min[i]);
        }
        if (r != -1) {
            T[x].max[i] = max(T[x].max[i], T[r].max[i]);
            T[x].min[i] = min(T[x].min[i], T[r].min[i]);
        }
    }
    // printf("x=%d; (%d %d) Tree: (%d %d %d)\n", x, T[x].l, T[x].r
    //     , T[x].A[0], T[x].A[1], T[x].A[2]);
}
#ifdef DEBUG_PROTECT
bool havefather[100007];
#endif
int build(Point* P, int size, int DIM)
{ // return root; build in cpu
    T = P;
    // for (int i = 0; i < size; i++)
    //     printf("T: %f %f %f %d %d\n", T[i].A[0], T[i].A[1], T[i].A[2], T[i].l, T[i].r);
    id = (int*)malloc(size * sizeof(int));
    for (int i = 0; i < size; i++)
        id[i] = i;
    int root;
    build_nodes(root, 0, size, 0, DIM);
    free(id);
    // for (int i = 0; i < size; i++)
    //     printf("T: %f %f %f %d %d\n",T[i].A[0],T[i].A[1],T[i].A[2],T[i].l,T[i].r);
    T = nullptr;
#ifdef DEBUG_PROTECT
    int cnt=0;
    for (int i = 0; i < size; i++)
        havefather[i] = 0;
    for (int i = 0; i < size; i++) {
        if (P[i].l != -1)
            havefather[P[i].l] = 1;
        if (P[i].r != -1)
            havefather[P[i].r] = 1;
    }
    for (int i = 0; i < size; i++)
        if (!havefather[i]) cnt++;
    if (cnt!=1) {
        printf("  ----------------     BUILD TREE WRONG! NULTI-ROOT!     -----------------  ");
    }
#endif
    //     printf("%d ", havefather[i]);
    // puts("");

    // for (int i = 0; i < size; i++)
    //     printf("P:id=%d %f %f %f %d %d (%f %f %f, %f %f %f)\n", i, P[i].A[0], P[i].A[1], P[i].A[2], P[i].l, P[i].r,
    //         P[i].min[0], P[i].min[1], P[i].min[2], P[i].max[0], P[i].max[1], P[i].max[2]);
    return root;
}

int build(float* point, int point_size, Point*& p, int dim = 3)
{
    p = (Point*)malloc(point_size * sizeof(Point));
    for (int i = 0; i < point_size; i++) {
        for (int k = 0; k < dim; k++)
            p[i].A[k] = point[i * dim + k];
        p[i].init();
        //print("%d %d %d\n", point[i * 3], point[i * 3+1], point[i * 3+2])
    }
    // for (int i = 0; i < point_size; i++)
    //     printf("p: (%f %f %f) %d %d; (%f %f %f, %f %f %f)\n", p[i].A[0], p[i].A[1], p[i].A[2], p[i].l, p[i].r,
    //         p[i].min[0], p[i].min[1], p[i].min[2], p[i].max[0], p[i].max[1], p[i].max[2]);
    return build(p, point_size, dim);
}

struct queryList {
    int qid, tid;
    queryList(int q, int t)
        : qid(q)
        , tid(t)
    {
    }
};
__device__ inline queryList pop(MyCudaArray<queryList>& Q, int& st,
    const int& startposition, const int& endposition)
{
    queryList ret = Q[st++];
    if (st == endposition)
        st = startposition;
    return ret;
}
__device__ inline void push(MyCudaArray<queryList>& Q, const int& st, int& ed, const queryList& now,
    const int& startposition, const int& endposition)
{
    if (st == ed + 1)
        printf("WARNING: ST == ED; PLEASE EXPAND THE POOL\n");
    Q[ed++] = now;
    if (ed == endposition)
        ed = startposition;
}
__device__ inline float getmaxdistance(const Point& now, const PointQuery& ini, const int dim)
{
    float maxdis = 0;
    for (int d = 0; d < dim; ++d) {
        if (now.min[d] > ini.A[d])
            maxdis += (now.min[d] - ini.A[d]) * (now.min[d] - ini.A[d]);
        else if (now.max[d] < ini.A[d])
            maxdis += (now.max[d] - ini.A[d]) * (now.max[d] - ini.A[d]);
        //printf("%f %f  max=%f\n", now.min[d], now.max[d], maxdis);
    }
    //printf("maxdis=%f\n",maxdis);
    return maxdis;
}
__device__ inline float getdistance(const Point& now, const PointQuery& ini, const int dim)
{
    float distance = 0;
    for (int d = 0; d < dim; ++d)
        distance += (now.A[d] - ini.A[d]) * (now.A[d] - ini.A[d]);
    return distance;
}

__global__ void Search(MyCudaArray<Point> P, MyCudaArray<int> result, MyCudaArray<int> pos, MyCudaArray<PointQuery> Query,
    MyCudaArray<queryList> QueryList, MyCudaArray<int> ST, MyCudaArray<int> ED, MyCudaArray<int> STARTPOSITION, int QUERYBLOCKSIZE,
    Lock mutex, MyCudaArray<int> running, float distance, const int dim, int THREADNUMBER)
{ //todo : change it to cycle; save should have block
    int id = START + BLOCKID * STRIDE, threadid = id % THREADNUMBER;
#ifdef DEBUG
    //printf("Search (%d+%d*%d) start; threadid=%d; start and end = (%d %d)\n", START, BLOCKID, STRIDE, threadid, ST[threadid], ED[threadid]);
#endif
    int startposition = STARTPOSITION[threadid];
    int endposition = startposition + QUERYBLOCKSIZE; // for dist check
    //mutex.lock();
    //printf("Q.size = %d (%d %d)\n", Q.size(), START, STRIDE);
    //mutex.unlock();
    //return;
    distance = distance * distance;
    while (true) {
        //mutex.lock();
        //printf("lock (%d %d) start\n", START, STRIDE);
        if (ST[threadid] == ED[threadid]) {
            //if (*running)
            //continue;
            //else {
            //mutex.unlock();
            break;
            //}
        }
        // printf("running = %d; mutex=%d(%d); %d(%d)\n", *running, mutex,
        //     *(mutex.mutex), threadid, ED[threadid] - ST[threadid]);
        //printf("before pop %d\n", Q.size());
        queryList nowquery = pop(QueryList, ST[threadid], startposition, endposition);
        Point now = P[nowquery.tid];
        PointQuery ini = Query[nowquery.qid];
        float nowdis = getdistance(now, ini, dim);
        // if (nowquery.pid == 1) {
        //float maxdis = getmaxdistance(now, ini, dim);
        //printf("searching %d %d; ST=%d; ED=%d;  xyz = (%f %f %f), l,r=(%d %d), ini=(%f %f %f);   dis=%f %f\n",
        //    nowquery.qid, nowquery.tid, ST[threadid], ED[threadid],
        //    now.A[0], now.A[1], now.A[2], now.l, now.r,
        //    ini.A[0], ini.A[1], ini.A[2], maxdis, nowdis);
        // }
        queryList newquery = nowquery;
        if (now.l != -1 && getmaxdistance(P[now.l], ini, dim) <= distance) {
            newquery.tid = now.l;
            push(QueryList, ST[threadid], ED[threadid], newquery, startposition, endposition);
        }
        if (now.r != -1 && getmaxdistance(P[now.r], ini, dim) <= distance) {
            newquery.tid = now.r;
            push(QueryList, ST[threadid], ED[threadid], newquery, startposition, endposition);
        }
        if (nowdis <= distance) {
            //printf("gotit:(push) qid,pid,pos,tid,nowdis=%d %d %d %f\n",nowquery.qid,pos[nowquery.qid], nowquery.tid, nowdis);
            result[pos[nowquery.qid]++] = nowquery.tid;
        }

        //printf("after pop %d\n", Q.size());
        //(*running)++;
        //mutex.unlock();
        /* check it */
        /* done */
        //mutex.lock();
        //(*running)--;
        //mutex.unlock();
    }
    //printf("%d: ED=%d\n", threadid, ED[threadid]);
    //printf("%d ", result.A);
    //printf("in cuda: %d %d %d\n", result[30], result[60], result[61]);
}
__host__ void search(Point* v, int point_size, float* query_points, int query_size,
    int maxcount, int* result, float distance, int root, const int dim) // from cpu; search_ids
{ // TODO: run ans metux(for multi process); querylist_size should change
#ifdef DEBUG
    double start = clock();
#endif
    int THREAD = query_size, BLOCK = 1;
    THREAD = std::min(THREAD, 512); //for data parallel
    int K = query_size / THREAD + (query_size % THREAD != 0), BLOCKSIZE = maxcount * K;
    //THREAD_limit: (query_size+K-1)/K * BLOCKSIZE * sizeof(queryList)
    MyCudaArray<Point> T(point_size);
    MyCudaArray<int> Ans(query_size * maxcount), cnt(query_size);
    MyCudaArray<PointQuery> QueryPoints(query_size);
    MyCudaArray<int> ST(query_size), ED(query_size), STARTPOSITION(query_size);
    MyCudaArray<queryList> QueryList(THREAD*BLOCKSIZE);
    T.toGPU(v);
#ifdef DEBUG
    printf("cuda malloc time = %f\n", (clock() - start) / CLOCKS_PER_SEC);
#endif
    // TODO: split it to multi_block
    //Ans.toGPU(result);
    // get the querylist
    int* cnt_cpu = (int*)malloc(query_size * sizeof(int));
    int *st_cpu = (int*)malloc(query_size * sizeof(int)), *ed_cpu = (int*)malloc(query_size * sizeof(int)), *startposition_cpu = (int*)malloc(query_size * sizeof(int));
    queryList* querylist_cpu = (queryList*)malloc(THREAD * BLOCKSIZE * sizeof(queryList)); //id;anspos
    PointQuery* querypoints_cpu = (PointQuery*)malloc(query_size * sizeof(PointQuery));//xyz;
    for (int i = 0; i < query_size; i++) {
        // ans_point and position
        int start_anspos = i * maxcount;
        cnt_cpu[i] = start_anspos;
        querypoints_cpu[i] = PointQuery();
        for (int k = 0; k < dim; k++)
            querypoints_cpu[i].A[k]=query_points[i * dim + k];
        //printf("%f %f %f\n", querypoints_cpu[i].A[0], querypoints_cpu[i].A[1], querypoints_cpu[i].A[2]);
        int block_inside = i / K, bias = i - block_inside * K;
        int block_start = block_inside * BLOCKSIZE + bias;
        if (bias == 0) {
            startposition_cpu[block_inside] = block_start;
            st_cpu[block_inside] = block_start;
        }
        querylist_cpu[block_start] = queryList(i, root);
        ed_cpu[block_inside] = block_start + 1;
    }
#ifdef DEBUG
    printf("cpu initialize time(real-time) = %f\n", (clock() - start) / CLOCKS_PER_SEC);
#endif
    cnt.toGPU(cnt_cpu); // it will be used later; no need to free
    ST.toGPU(st_cpu);
    ED.toGPU(ed_cpu);
    STARTPOSITION.toGPU(startposition_cpu);
    QueryPoints.toGPU(querypoints_cpu);
    QueryList.toGPU(querylist_cpu);
    free(st_cpu);
    free(ed_cpu);
    free(querypoints_cpu);
    free(startposition_cpu);
    free(querylist_cpu);
#ifdef DEBUG
    printf("query_size = %d\n", query_size);
#endif
#ifdef DEBUG
    printf("cuda initialize time(real-time) = %f\n", (clock() - start) / CLOCKS_PER_SEC);
#endif
    //printf("build done ???");
    //printf("BLOCK, STRIDE = %d %d; K=%d; blocksiez=%d\n", BLOCK, THREAD, K, BLOCKSIZE);

    //multi-thread-for-one-BLOCK(todo)
    BLOCK = std::max(THREAD_BLOCKS / THREAD, 1);
    BLOCK = std::min(BLOCK, 1); //BLOCK=1; not it is not used
    Lock mutex(BLOCK);
    MyCudaArray<int> run(BLOCK);
    //RESTRUCT THREAD
    int all = BLOCK * THREAD;
    int real_block = std::max(BLOCK, std::min(16, all));
    int real_thread = all / real_block;
#ifdef DEBUG
    double _t=clock();
#endif
    Search<<<real_block, real_thread>>>(T, Ans, cnt, QueryPoints,
        QueryList, ST, ED, STARTPOSITION, BLOCKSIZE, mutex, run, distance, dim, THREAD);
    Ans.toCPU(result);
#ifdef DEBUG
    printf("cuda query time(real-time) = %f;   CUDA_TIME = %f\n", (clock() - start) / CLOCKS_PER_SEC, (clock()-_t) / CLOCKS_PER_SEC);
#endif
#ifdef DEBUG_PROTECT
    int max_ans_count = -0x3f3f3f3f, min_ans_count = 0X3f3f3f3f, sum_ans_count=0;
    int minvalue = 0x3f3f3f3f, maxvalue = -0X3f3f3f3f;
#endif
    cnt.toCPU(cnt_cpu);
    for (int i = 0; i < query_size; i++) {
        int start_anspos = i * maxcount;
        int nowcount = cnt_cpu[i] - start_anspos;
        if (nowcount > maxcount)
            printf("WRONG! query %d num_points_inside > maxcount\n", i);
        std::random_shuffle(result + start_anspos, result + cnt_cpu[i]);
        for (int k = cnt_cpu[i]; k < start_anspos + maxcount; k++) {
            result[k] = result[k - nowcount];
            if (!nowcount) result[k]=0;
        }
#ifdef DEBUG_PROTECT
        max_ans_count = max(max_ans_count, nowcount);
        min_ans_count = min(min_ans_count, nowcount);
        sum_ans_count += nowcount;
#ifdef DEBUG
        int noww=0;
        for (int k=0; k<point_size; k++)
            if ((v[k].A[0]-query_points[i*3+0])*(v[k].A[0]-query_points[i*3+0])+
                (v[k].A[1]-query_points[i*3+1])*(v[k].A[1]-query_points[i*3+1])+
                (v[k].A[2]-query_points[i*3+2])*(v[k].A[2]-query_points[i*3+2])<=distance*distance) noww++;
        if (noww!=nowcount)
            printf("WRONG! CNT NOT RIGHT!");
#endif
        for (int k = start_anspos; k < start_anspos + maxcount; k++) {
            minvalue = min(minvalue, result[k]), maxvalue = max(maxvalue, result[k]);
            if (result[k]>100000||result[k]<-100000) {
                printf(" WRONG!!!  result[%d] out of range; pos=(%d,%d); val=%d; cnt=%d\n",k,i,k-start_anspos,result[k],min_ans_count);
            }
        }
#endif
        //printf("from %d to %d\n",cnt_cpu[i],start_anspos+maxcount);
    }
#ifdef DEBUG_PROTECT
    if (min_ans_count==0)
        printf("           ===== max Ans count = %d (mincount=%d; mean=%f value=%d %d); query_size=%d =====\n", max_ans_count, min_ans_count, 1.*sum_ans_count/query_size, minvalue, maxvalue, query_size);
#ifdef DEBUG
    printf("           ===== max Ans count = %d (mincount=%d; mean=%f value=%d %d) =====\n", max_ans_count, min_ans_count, 1.*sum_ans_count/query_size, minvalue, maxvalue);
#endif
#endif
    free(cnt_cpu);
    //printf("%d ", Ans.A);
    //printf("%d %d %d\n", result[30], result[60], result[61]);
    T.Free();
    Ans.Free();
    // query
    QueryPoints.Free();
    cnt.Free();
    ST.Free();
    ED.Free();
    // queue
    STARTPOSITION.Free();
    QueryList.Free();
    mutex.Free();
    run.Free();
    //printf("search done ???\n");
}
}
void SearchRadius(float* point, int point_size, float* query, int query_size, float distance, int max_count, int* result, int dim = 3)
{
#ifdef DEBUG
    printf("dist, max_count = %f %d (%d (%d) %d)\n", distance, max_count, max_count/2, max_count/4, max_count/8);
    double start = clock();
#endif
    KDTrees::Point* p;
    int root = KDTrees::build(point, point_size, p, dim);
#ifdef DEBUG
    printf("build time = %f\n", (clock() - start) / CLOCKS_PER_SEC);
    start = clock();
#endif
    //for (int i = 0; i < point_size; i++)
    //    printf("p: id=%d (%f %f %f) %d %d; (%f %f %f, %f %f %f)\n", i, p[i].A[0], p[i].A[1], p[i].A[2], p[i].l, p[i].r,
    //        p[i].min[0], p[i].min[1], p[i].min[2], p[i].max[0], p[i].max[1], p[i].max[2]);
    KDTrees::search(p, point_size, query, query_size, max_count, result, distance, root, dim);
    free(p);
#ifdef DEBUG
    printf("search time = %f\n", (clock() - start) / CLOCKS_PER_SEC);
#endif
}
#endif //_KD_TEEE_HPP_
