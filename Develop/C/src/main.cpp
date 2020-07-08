#include "SkyNet.h"

static layer config[layer_count] = {
{ "conv0", 320,160,4,  320,160,4,  0,0,0 },  //conv0
{ "conv1", 320,160,32, 320,160,32, 3,1,1 },  //conv1
{ "conv2", 320,160,32, 320,160,64, 1,1,0 },  //conv2
{ "pool1", 320,160,64, 160,80,64,  2,2,0 },  //pool1
{ "conv3", 160,80,64,  160,80,64,  3,1,1 },  //conv3
{ "conv4", 160,80,64,  160,80,96,  1,1,0 },  //conv4
{ "pool2", 160,80,96,  80,40,96,   2,2,0 },  //pool2
{ "conv5", 80,40,96,   80,40,96,   3,1,1 },  //conv5
{ "conv6", 80,40,96,   80,40,192,  1,1,0 },  //conv6
{ "reorg", 80,40,192,  40,20,768,  2,2,0 },  //reorg
{ "pool3", 80,40,192,  40,20,192,  2,2,0 },  //pool3
{ "conv7", 40,20,192,  40,20,192,  3,1,1 },  //conv7
{ "conv8", 40,20,192,  40,20,384,  1,1,0 },  //conv8
{ "conv9", 40,20,384,  40,20,384,  3,1,1 },  //conv9
{ "conv10",40,20,384,  40,20,512,  1,1,0 },  //conv10
{ "cat",   40,20,192,  40,20,1280, 0,0,0 },  //concat
{ "conv11",40,20,1280, 40,20,1280, 3,1,1 },  //conv11
{ "conv12",40,20,1280, 40,20,96,   1,1,0 },  //conv12
{ "conv13",40,20,96,   40,20,32,   1,1,0 },  //conv13
};

float anchor[4]  = {1.4940052559648322, 2.3598481287086823, 4.0113013115312155, 5.760873975661669};
float bbox_m[10] = {52., 48., 28., 30., 124., 47., 52., 23., 23., 125.};

float sigmoid(float x)
{
	return 1/(1+exp(-x));
}

void Compute_BBOX(BDT16* BBOX)
{
#define h 20
#define w 40
    BDT bbox_origin[4][7];
	float bbox[4][4];
	for(int b=0; b<4; b++)
	{
		bbox_origin[b][0] = BBOX[b].range(15,0);  //xs
		bbox_origin[b][1] = BBOX[b].range(31,16); //ys
		bbox_origin[b][2] = BBOX[b].range(47,32); //ws
		bbox_origin[b][3] = BBOX[b].range(63,48); //hs
		bbox_origin[b][4] = BBOX[b].range(79,64); //flag
		bbox_origin[b][5] = BBOX[b].range(95,80); //x
		bbox_origin[b][6] = BBOX[b].range(112,96);//y
	}
	for(int b=0; b<4; b++)
	{	
		if(bbox_origin[b][4]>0)
		{
			float xs = bbox_origin[b][0]*bbox_m[5]/qm;
			float ys = bbox_origin[b][1]*bbox_m[6]/qm;
			float ws = bbox_origin[b][2]*bbox_m[7]/qm;
			float hs = bbox_origin[b][3]*bbox_m[8]/qm;
			float xs_inb = sigmoid(xs) + bbox_origin[b][5];
			float ys_inb = sigmoid(ys) + bbox_origin[b][6];
			float ws_inb = exp(ws)*anchor[2];
			float hs_inb = exp(hs)*anchor[3];
			float bcx = xs_inb/w;
			float bcy = ys_inb/h;
			float bw = ws_inb/w;
			float bh = hs_inb/h;
			bbox[b][0] = bcx - bw/2.0; //xmin
			bbox[b][1] = bcy - bh/2.0; //ymin
			bbox[b][2] = bcx + bw/2.0; //xmax
			bbox[b][3] = bcy + bh/2.0; //ymax
		}
		else
		{
			float xs = bbox_origin[b][0]*bbox_m[0]/qm;
			float ys = bbox_origin[b][1]*bbox_m[1]/qm;
			float ws = bbox_origin[b][2]*bbox_m[2]/qm;
			float hs = bbox_origin[b][3]*bbox_m[3]/qm;
			float xs_inb = sigmoid(xs) + bbox_origin[b][5];
			float ys_inb = sigmoid(ys) + bbox_origin[b][6];
			float ws_inb = exp(ws)*anchor[0];
			float hs_inb = exp(hs)*anchor[1];
			float bcx = xs_inb/w;
			float bcy = ys_inb/h;
			float bw = ws_inb/w;
			float bh = hs_inb/h;
			bbox[b][0] = bcx - bw/2.0; //xmin
			bbox[b][1] = bcy - bh/2.0; //ymin
			bbox[b][2] = bcx + bw/2.0; //xmax
			bbox[b][3] = bcy + bh/2.0; //ymax
		}
	}
	for(int b=0; b<4; b++)
	{
		printf("img %d xmin: %f, ymin: %f, xmax: %f, ymax: %f\n", b, bbox[b][0], bbox[b][1], bbox[b][2], bbox[b][3]);
	}
}

int main() {

//*************************************init *********************************
	printf("init SkyNet \n");
	WDT32* weight;
	BDT16* biasm;
	ADT4* img;
	ADT* data[4];
	ADT* ofm_blob;
	ADT32* ofm_blob32;
	ADT* ofm[4];

	for(int p=0; p<4; p++)
	{
		data[p] = (ADT*)sds_alloc(32*160*320*sizeof(ADT));
		ofm[p] = (ADT*)sds_alloc(64*320*640*sizeof(ADT));
	}
	img = (ADT4*)sds_alloc(4*160*320*sizeof(ADT4));
	weight = (WDT32*)sds_alloc(441344*sizeof(WDT));
	biasm = (BDT16*)sds_alloc(432*sizeof(BDT16));
	ofm_blob32 = (ADT32*)sds_alloc(32*fm_all*sizeof(ADT));
	ofm_blob = (ADT*)sds_alloc(64*643*323*sizeof(ADT));
	//*************************************load data *********************************
	printf("load parameter\n");
	load_weight(weight, 441344);
	load_biasm(biasm, 6848);

	printf("load image\n");
	for (int b = 0; b<4; b++)
	{
		load_fm(data[b], config[0]);
		img_DT_2_DT4(data[b], img, config[0], b);
	}
	//*************************************HLS, Skynet *********************************
	printf("SkyNet start\n");
	timeval start,end;
	gettimeofday(&start, NULL);
	SkyNet(img, ofm_blob32, weight, biasm);
	gettimeofday(&end, NULL);
	printf("SkyNet costs %luus\n", (end.tv_sec-start.tv_sec)*1000000+end.tv_usec-start.tv_usec);

	//************************************* check the results*********************************
	printf("check SkyNet results \n");

	fm_DT32_2_DT(&ofm_blob32[pool1_o], ofm_blob, config[3]);
	distitch(ofm_blob, ofm, config[3]);
	for (int p = 0; p<4; p++)
	{
		check_fm(ofm[p], config[3]);
	}

	fm_DT32_2_DT(&ofm_blob32[pool2_o], ofm_blob, config[6]);
	distitch(ofm_blob, ofm, config[6]);
	for (int p = 0; p<4; p++)
	{
		check_fm(ofm[p], config[6]);
	}

	fm_DT32_2_DT(&ofm_blob32[conv5_o], ofm_blob, config[7]);
	distitch(ofm_blob, ofm, config[7]);
	for (int p = 0; p<4; p++)
	{
		check_fm(ofm[p], config[7]);
	}

	fm_DT32_2_DT(&ofm_blob32[conv6_o], ofm_blob, config[8]);
	distitch(ofm_blob, ofm, config[8]);
	for (int p = 0; p<4; p++)
	{
		check_fm(ofm[p], config[8]);
	}

	fm_DT32_2_DT(&ofm_blob32[pool3_o], ofm_blob, config[10]);
	distitch(ofm_blob, ofm, config[10]);
	for (int p = 0; p<4; p++)
	{
		check_fm(ofm[p], config[10]);
	}

	fm_DT32_2_DT(&ofm_blob32[conv7_o], ofm_blob, config[11]);
	distitch(ofm_blob, ofm, config[11]);
	for (int p = 0; p<4; p++)
	{
		check_fm(ofm[p], config[11]);
	}

	fm_DT32_2_DT(&ofm_blob32[conv8_o], ofm_blob, config[12]);
	distitch(ofm_blob, ofm, config[12]);
	for (int p = 0; p<4; p++)
	{
		check_fm(ofm[p], config[12]);
	}

	fm_DT32_2_DT(&ofm_blob32[conv9_o], ofm_blob, config[13]);
	distitch(ofm_blob, ofm, config[13]);
	for (int p = 0; p<4; p++)
	{
		check_fm(ofm[p], config[13]);
	}

	fm_DT32_2_DT(&ofm_blob32[conv10_o], ofm_blob, config[14]);
	distitch(ofm_blob, ofm, config[14]);
	for (int p = 0; p<4; p++)
	{
		check_fm(ofm[p], config[14]);
	}

	fm_DT32_2_DT(&ofm_blob32[conv11_o], ofm_blob, config[16]);
	distitch(ofm_blob, ofm, config[16]);
	for (int p = 0; p<4; p++)
	{
		check_fm(ofm[p], config[16]);
	}

	fm_DT32_2_DT(&ofm_blob32[conv12_o], ofm_blob, config[17]);
	distitch(ofm_blob, ofm, config[17]);
	for (int p = 0; p<4; p++)
	{
		check_fm(ofm[p], config[17]);
	}
	
	Compute_BBOX(&biasm[bbox_o]);

	for(int p=0; p<4; p++)
	{
		sds_free(data[p]);
		sds_free(ofm[p]);
	}
	sds_free(weight);
	sds_free(biasm);
	sds_free(ofm_blob32);
	sds_free(ofm_blob);
}
