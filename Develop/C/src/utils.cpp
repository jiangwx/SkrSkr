#include "SkyNet.h"

void generate_fm(DT* fm, layer l)
{
    for(int c=0; c<l.oc; c++)
    {
        for(int h=0; h<l.oh; h++)
        {
            for(int w=0; w<l.ow; w++)
            {
                int fm_index = c*l.oh*l.ow + h*l.ow + w;
                fm[fm_index] = h + w;
            }
        }
    }
}

void check(DT* result, DT* golden, int len, layer l)
{
    int err = 0;
    for (int j = 0; j < len; j++)
    {
        if (((result[j] - golden[j]) > check_scale) || ((result[j] - golden[j]) < -check_scale))
        {
            err++;
            //printf("[%d] correct=%f,wrong=%f\n", j, tmp[j], fm[j]);
        }
    }

    if (err > 0)
        printf("%s error cnt= %d\n", l.name, err);
    else
        printf("%s correct \n", l.name);
}

void load_fm(ADT* fm, layer l)
{
    char nstr[50];

    sprintf(nstr, "./blob/%s.bb", l.name);
    FILE *fp = fopen(nstr, "rb");
    fread(fm, 1, l.ow*l.oh*l.oc * sizeof(ADT), fp);
    fclose(fp);
}

void load_weight(WDT32* weight , int length)
{
    char nstr[50];
    sprintf(nstr, "./weight/SkyNet.wt");
    FILE *fp = fopen(nstr, "rb");
    fread(weight, 1, length*sizeof(WDT), fp);
    fclose(fp);
}
void load_biasm(BDT16* biasm , int length)
{
    char nstr[50];
    sprintf(nstr, "./weight/SkyNet.bm");
    FILE *fp = fopen(nstr, "rb");
    fread(biasm, 1, length*sizeof(BDT), fp);
    fclose(fp);
}

void load_bias(DT* bias , int length, layer l)
{
    char nstr[50];
    sprintf(nstr, "./weight/%s.bs", l.name);
    FILE *fp = fopen(nstr, "rb");
    fread(bias, 1, length*sizeof(DT), fp);
    fclose(fp);
}

void show_fm(ADT* fm, layer l)
{
    for (int c=0;c<l.oc;c++)
    {
        for (int h=0;h<l.oh;h++)
        {
            for (int w=0;w<l.ow;w++)
            {
                int i = c*l.oh*l.ow + h*l.ow + w;
                std::cout << fm[i]<<", ";
            }
            std::cout << "\n";
        }
        std::cout << "\n";
    }
}

void check_fm(ADT* fm, layer l)
{
    int len = l.oc*l.ow*l.oh;
    ADT *tmp = (ADT *)malloc(sizeof(ADT)*len);

    char nstr[50];
    sprintf(nstr, "./blob/%s.bb", l.name);
    FILE *fp = fopen(nstr, "rb");
    fread(tmp, 1, len*sizeof(ADT), fp);
    fclose(fp);

    int err = 0;
    int zero;
    for(int c=0; c<l.oc; c++)
    {
        for(int h=0; h<l.oh; h++)
        {
            for(int w=0; w<l.ow; w++)
            {
                int index = c*l.oh*l.ow + h*l.ow + w;
                if (fm[index]!=tmp[index])
                {
                    err++;
                }
            }
        }
    }

    if (err > 0)
        printf("%s error cnt= %d\n", l.name, err);
    else
        printf("%s correct \n", l.name);

    free(tmp);
}

void check_bbox(BDT* bbox, layer l)
{
    int len = l.oc*l.ow*l.oh;
    BDT *tmp = (BDT*)malloc(sizeof(BDT)*len);

    char nstr[50];
    sprintf(nstr, "./blob/%s.bb", l.name);
    FILE *fp = fopen(nstr, "rb");
    fread(tmp, 1, len*sizeof(BDT), fp);
    fclose(fp);

    int err = 0;
    int zero;
    for(int c=0; c<16; c++)
    {
        for(int h=0; h<l.oh; h++)
        {
            for(int w=0; w<l.ow; w++)
            {
                int index = c*l.oh*l.ow + h*l.ow + w;
                if (bbox[index]!=tmp[index])
                {
                    err++;
                }                
            }
        }
    }
    if (err > 0)
        printf("%s error cnt= %d\n", l.name, err);
    else
        printf("%s correct \n", l.name);

    free(tmp);
}

