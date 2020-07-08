#ifndef SKYNET_H
#define SKYNET_H

#include <stdlib.h>
#include <stdio.h>
#include <iostream>
#include <vector>
#include <memory.h>
#include <time.h>
#include <sys/time.h>
#include <fstream>
#include <cstring>
#include <math.h>
#include "ap_int.h"

#ifdef __SDSCC__
#include "sds_lib.h"
#else
#define sds_alloc malloc
#define sds_free free
#endif

#define __AP_INT__
#ifdef __AP_INT__
typedef ap_uint<8> ADT;
typedef ap_int<19> RDT;
typedef ap_int<16> BDT;
typedef ap_int<6>  WDT;
typedef ap_int<16> MDT;
typedef ap_int<256> WDT32;
typedef ap_int<32>  ADT4;
typedef ap_int<256> ADT32;
typedef ap_int<256> BDT16;
typedef ap_int<256> MDT16;
#else
typedef int DT;
typedef unsigned char ADT;
typedef short BDT;
typedef char WDT;
typedef short MDT;
typedef int RDT;
typedef ap_int<256> WDT32;
typedef ap_int<32>  ADT4;
typedef ap_int<256> ADT32;
typedef ap_int<256> BDT16;
typedef ap_int<256> MDT16;
#endif

#define na 8
#define nw 6
#define nb 16
#define nm 17
#define qm 131072.0
typedef int DT;

#define amin 0
#define amax 255
#define bmin -32768
#define bmax 32767
#define rmax 262143
#define rmin -262144

#define layer_count 19
#define check_scale 0.00001

struct layer
{
	char name[10];
	int iw, ih, ic, ow, oh, oc;
	int k, s, p;
};

#define pool1_o  0
#define pool2_o  105298
#define conv5_o  145885
#define conv6_o  186472
#define pool3_o  267646
#define conv7_o  289060
#define conv8_o  331888
#define conv9_o  374716
#define conv10_o 417544
#define conv11_o 474648
#define conv12_o 617408
#define fm_all   628115

#define conv1_b 0
#define conv2_b 4
#define conv3_b 12
#define conv4_b 20
#define conv5_b 32
#define conv6_b 44
#define conv7_b 68
#define conv8_b 92
#define conv9_b 140
#define conv10_b 188
#define conv11_b 252
#define conv12_b 412
#define conv13_b 424

#define conv1_m 2
#define conv2_m 8
#define conv3_m 16
#define conv4_m 26
#define conv5_m 38
#define conv6_m 56
#define conv7_m 80
#define conv8_m 116
#define conv9_m 164
#define conv10_m 220
#define conv11_m 332
#define conv12_m 418
#define conv13_m 426
#define bbox_o 428

#define conv1_w 0
#define conv2_w 9
#define conv3_w 73
#define conv4_w 91
#define conv5_w 283
#define conv6_w 310
#define conv7_w 886
#define conv8_w 940
#define conv9_w 3244
#define conv10_w 3352
#define conv11_w 9496
#define conv12_w 9856
#define conv13_w 13696

/**********utils.cpp************/
void load_fm(ADT* fm, layer l);
void load_weight(WDT32 *weight, int length);
void load_biasm(BDT16* biasm , int length);
void check(ADT* result, ADT* golden, int len, layer l);
void check_fm(ADT* fm, layer l);
void check_bbox(BDT* bbox, layer l);
void show_fm(ADT* fm, layer l);

void generate_fm(DT* fm, layer l);
void generate_weight(DT* weight, layer l);

/**********transform.cpp************/
void stitch(ADT* ifm[4], ADT* ofm, layer l);
void distitch(ADT* ifm, ADT* ofm[4], layer l);
void img_DT_2_DT4(ADT* in, ADT4* out, layer l, int b);
void img_DT_2_DT3(ADT* in, ADT* out, layer l, int b);
void fm_DT_2_DT32(ADT* in, ADT32* out, layer l);
void fm_DT32_2_DT(ADT32* in, ADT* out, layer l);
void distitch_bbox(BDT* ifm, BDT* ofm[4], layer l);
void bbox_DT16_2_DT(BDT16* in, BDT* out, layer l);

/**********SkyNet.h [HW]************/
void SkyNet(ADT4* img, ADT32* fm, WDT32* weight, BDT16* biasm);
#endif
