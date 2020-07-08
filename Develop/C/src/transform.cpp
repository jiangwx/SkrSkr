#include "SkyNet.h"

void stitch(ADT* ifm[4], ADT* ofm, layer l)
{   
    int offset_h[4];
    offset_h[0] = offset_h[1] = 1;
    offset_h[2] = offset_h[3] = l.oh + 2;
    int offset_w[4];
    offset_w[0] = offset_w[2] = 1;
    offset_w[1] = offset_w[3] = l.ow + 2;
    for(int i=0; i<l.oc*(l.oh*2+3)*(l.ow*2+3); i++)
    {
        ofm[i] = 128;
    }
    for(int p=0; p<4; p++)
    {
        for(int c=0; c<l.oc; c++)
        {
            for(int h=0; h<l.oh; h++)
            {
                for(int w=0; w<l.ow; w++)
                {
                    int ifm_index = c*l.oh*l.ow + h*l.ow + w;
                    int ofm_index = c*(l.oh*2+3)*(l.ow*2+3) + (h+offset_h[p])*(l.ow*2+3) + (w+offset_w[p]);
                    ofm[ofm_index] = ifm[p][ifm_index];
                }
            }
        }
    }
}

void distitch(ADT* ifm, ADT* ofm[4], layer l)
{   
    int offset_h[4];
    offset_h[0] = offset_h[1] = 1;
    offset_h[2] = offset_h[3] = l.oh + 2;
    int offset_w[4];
    offset_w[0] = offset_w[2] = 1;
    offset_w[1] = offset_w[3] = l.ow + 2;

    for(int p=0; p<4; p++)
    {
        for(int c=0; c<l.oc; c++)
        {
            for(int h=0; h<l.oh; h++)
            {
                for(int w=0; w<l.ow; w++)
                {
                    int ifm_index = c*(l.oh*2+3)*(l.ow*2+3) + (h+offset_h[p])*(l.ow*2+3) + (w+offset_w[p]);
                    int ofm_index = c*l.oh*l.ow + h*l.ow + w;
                    ofm[p][ofm_index] = ifm[ifm_index];
                }
            }
        }
    }
}


void fm_DT_2_DT32(ADT* in, ADT32* out, layer l)
{
	for (int Mx = 0; Mx < l.oc / 32; Mx++)
	{
		for (int i = 0; i < (2*l.oh+3)*(2*l.ow+3); i++)
		{
			for (int tm = 0; tm < 32; tm++)
			{
				out[Mx*(2*l.oh+3)*(2*l.ow+3)+i].range(8*tm+7,8*tm)=in[(tm+Mx*32)*(2*l.oh+3)*(2*l.ow+3)+i];
			}
		}
	}
}

void img_DT_2_DT4(ADT* in, ADT4* out, layer l, int b)
{
	for (int i=0; i<160*320; i++)
	{
		for (int tm=0; tm<4; tm++)
		{
			out[b*320*160+i].range(8*tm+7,8*tm)=in[tm*160*320+i];
		}
	}
}

void img_DT_2_DT3(ADT* in, ADT* out, layer l, int b)
{
	for (int i=0; i<160*320; i++)
	{
		for (int tm=0; tm<3; tm++)
		{
			out[(b*320*160+i)*3+tm]=in[tm*160*320+i];
		}
	}
}

void fm_DT32_2_DT(ADT32* in, ADT* out, layer l)
{
	for (int Mx = 0; Mx < l.oc / 32; Mx++)
	{
		for (int i = 0; i < (2*l.oh+3)*(2*l.ow+3); i++)
		{
			for (int tm = 0; tm < 32; tm++)
			{
				out[(tm+Mx*32)*(2*l.oh+3)*(2*l.ow+3) + i] = in[Mx*(2*l.oh+3)*(2*l.ow+3)+i].range(8*tm+7,8*tm);
			}
		}
	}
}

void distitch_bbox(BDT* ifm, BDT* ofm[4], layer l)
{   
    int offset_h[4];
    offset_h[0] = offset_h[1] = 1;
    offset_h[2] = offset_h[3] = l.oh + 2;
    int offset_w[4];
    offset_w[0] = offset_w[2] = 1;
    offset_w[1] = offset_w[3] = l.ow + 2;

    for(int p=0; p<4; p++)
    {
        for(int c=0; c<16; c++)
        {
            for(int h=0; h<l.oh; h++)
            {
                for(int w=0; w<l.ow; w++)
                {
                    int ifm_index = c*(l.oh*2+3)*(l.ow*2+3) + (h+offset_h[p])*(l.ow*2+3) + (w+offset_w[p]);
                    int ofm_index = c*l.oh*l.ow + h*l.ow + w;
                    ofm[p][ofm_index] = ifm[ifm_index];
                }
            }
        }
    }
}

void bbox_DT16_2_DT(BDT16* in, BDT* out, layer l)
{
	for (int i = 0; i < (2*l.oh+3)*(2*l.ow+3); i++)
	{
		for (int tm = 0; tm<16; tm++)
		{
			out[tm*(2*l.oh+3)*(2*l.ow+3) + i] = in[i].range(16*tm+15,16*tm);
		}
	}
}

