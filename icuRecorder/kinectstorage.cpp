#include "kinectstorage.h"
#include <vector>
#include <stdio.h>
#include <cstring>



KinnectStorage::KinnectStorage(std::string fname,mode mod,int fps)
{

    this->fps = fps;
    lenght=0;
    KinnectStorage::header head;
    head.fps=fps;
    head.lenght=lenght;
    head.offset=0;
    curpos=0;

    outfile=0;
    infile =0;

    /* // Maybe resampling of depth data is needed
    for (int i=0; i<2048; i++) {
           float v = i/2048.0;
           v = powf(v, 3)* 6;
           t_gamma[i] = v*6*256;
    }*/
    positions.clear();


    switch(mod)
    {
        case KinnectStorage::CLOSED:
            break;
        case KinnectStorage::WRITE:
            curmod = WRITE;           
            outfile =new std::ofstream(fname.c_str(), std::ios::out | std::ios::binary);
            outfile->write((const char *)&head,sizeof(KinnectStorage::header));
            break;
        case KinnectStorage::READ:
            //Reading header
            curmod = READ;
            infile =new std::ifstream(fname.c_str(), std::ios::in | std::ios::binary);
            infile->read((char *)&head,sizeof(KinnectStorage::header));      
            fps = head.fps;
            lenght = head.lenght;
            //Reading positions
            infile->seekg(head.offset);
            size_t dsz=0;
            infile->read((char *)&dsz, sizeof(size_t));
            positions.resize(dsz);
            infile->read((char *)&positions.front(), sizeof(uint32_t)*dsz);
            //Seek back to the begin
            infile->seekg(sizeof(KinnectStorage::header));
            break;

    }
}
bool KinnectStorage::split16BitFrame(Mat &src,Mat &dst1,Mat &dst2)
{
    MatIterator_<ushort> it = src.begin<ushort>();
    MatIterator_<ushort> it_end = src.end<ushort>();
    MatIterator_<uchar> it2 = dst1.begin<uchar>();
    MatIterator_<uchar> it3 = dst2.begin<uchar>();

    for(; it != it_end; ++it)
    {
        int pval = *it;
        int lb = pval & 0xff;
        *it2 = saturate_cast<uchar>(pval>>8);
        *it3 = saturate_cast<uchar>(lb);
        ++it2;
        ++it3;
    }

    return true;
}

bool KinnectStorage::merge16BitFrame(Mat &src1,Mat &src2,Mat &dest)
{
    if(0==dest.data)
    {
        dest.create(cv::Size(src1.cols,src1.rows),CV_16UC1);
    }
    MatIterator_<uchar> it = src1.begin<uchar>();
    MatIterator_<uchar> it_end = src1.end<uchar>();
    MatIterator_<uchar> it2 = src2.begin<uchar>();

    MatIterator_<ushort> it3 = dest.begin<ushort>();

    for(; it != it_end; ++it)
    {


        int z = *it<<8;
        int lb = *it2;
        *it3 = saturate_cast<ushort>(z+lb);

        ++it2;
        ++it3;

    }

    return true;
}
inline void KinnectStorage::writeImage(char* ext,const Mat& img)
{
    static std::vector<uchar> data;
    if(cv::imencode(ext,img,data))
    {
        std::size_t dsz = data.size();
        outfile->write((const char *)&dsz, sizeof(size_t));
        outfile->write((const char *)&data.front(), data.size()*sizeof(uchar));
    }
}
inline Mat KinnectStorage::readImage()
{
    size_t dsz=0;
    static std::vector<uchar> data;
    data.clear();
    infile->read((char *)&dsz, sizeof(size_t));
    data.resize(dsz);
    infile->read((char *)&data.front(), sizeof(uchar)*dsz);
    Mat dp(data);

    return cv::imdecode(dp,-1);
}

bool KinnectStorage::write(uint32_t ts,Mat rgb,Mat depth)
{
    if(!outfile||(curmod!=WRITE))
    {
        return false;
    }

    positions.push_back(outfile->tellp());

    static Mat dst1(Size(640,480),CV_8UC1);
    static Mat dst2(Size(640,480),CV_8UC1);

    outfile->write((const char *)&ts,sizeof(uint32_t));

    split16BitFrame(depth,dst1,dst2);
    writeImage(".png",dst1);
    writeImage(".png",dst2);
    writeImage(".jpg",rgb);
    lenght++;
    return true;
}

bool KinnectStorage::read(Mat &rgb,Mat &depth,long pos)
{
    if(!infile||(curmod!=READ))
    {
        return false;
    }


    uint32_t ts;

    if(pos>=0)
    {
        curpos=pos;
        infile->seekg(positions[curpos]);
    }
    if(curpos>=length())
    {
        return false;
    }

    infile->read((char *)&ts, sizeof(uint32_t));


    static Mat src1(Size(640,480),CV_8UC1);
    static Mat src2(Size(640,480),CV_8UC1);

    src1 = readImage();
    src2 = readImage();
    rgb = readImage();

    this->merge16BitFrame(src1,src2,depth);
    curpos++;
    return true;
}

uint32_t KinnectStorage::length()
{
    return lenght;
}

KinnectStorage::~KinnectStorage()
{
    close();
}
void KinnectStorage::close()
{
    curmod = CLOSED;
    if(outfile)
    {
        KinnectStorage::header head;
        head.offset = outfile->tellp();
        //Saving frames positions
        std::size_t dsz = positions.size();
        outfile->write((const char *)&dsz, sizeof(size_t));
        outfile->write((const char *)&positions.front(), positions.size()*sizeof(uint32_t));
        //Rewrite header
        outfile->seekp(0);
        head.fps=fps;
        head.lenght=lenght;
        outfile->write((const char *)&head,sizeof(KinnectStorage::header));
        //Close the stream
        delete outfile;
        outfile=NULL;
    }
    if(infile)
    {
        delete infile;
        infile=NULL;
    }
}
