#ifndef KINNECTSTORAGE_H
#define KINNECTSTORAGE_H
#include <string>
#include <fstream>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

using namespace cv;

/*! Class KinnectStorage store compressed data from kinnetsensor
 * into file Depth inforamtion if compressed lossless using PNG
 *algorithm and RGB data is compressed using JPEG algorithm.
 * Compessed data weight is about 5Mb/sec
 */
class KinnectStorage
{
    /*! File header:
      * lenght - number of frames
      * fps - frames par second
      */
    struct header{
        uint32_t lenght;
        int fps;
        uint32_t offset;
    };

public:
    /*! File open mode
      */
    enum mode{
        CLOSED,//!< File is closed
        READ,//!< Reading of file
        WRITE //!< Writing file
    };
    /*! Creation of kinnectStorage
      \param fname - the name of the file
      \param mod - reading/writing mode
      \param fps = frames par second
      */
    KinnectStorage(std::string fname,mode mod=READ,int fps=25);
//    KinnectStorage(std::string fname,mode mod=READ,int fps=30);    
    /*! Write in file compressed into stream
      * \param ts - timestamp
      * \param rgb - color image
      * \param depth - the depth image from kinnect sensor
      * \return true is succeeded. false otherwise
      */
    bool write(uint32_t ts,Mat rgb,Mat depth);
    /*! Read a frame from opened storage
    * \param rgb - reference to color image (May be not initiated)
    * \param depth - reference to depth image from kinnect sensor (May be not initiated)
    * \param pos - position of the image infile for 0 to length-1
    * not implemented yet. -1 means to take next frame.
    * \return true is succeeded. false otherwise
    */
    bool read(Mat &rgb,Mat &depth,long pos=-1);
    /*! Return file length
      */
    uint32_t length();
    /*! Deallocate memory save headers into file
      */
    ~KinnectStorage();

    void close();

private:
    /*! Slipts 16bit image (depth image) into two 8bit image
      upper byte and lower byte
      \param src - 16bit source image
      \param dst1 - 8bit image upper byte
      \param dst2 - 8bit image lower byte
      \return true is succeeded. false otherwise
      */
    bool split16BitFrame(Mat &src,Mat &dst1,Mat &dst2);
    /*! Merge into 16bit image (depth image) from two 8bit image
      upper byte and lower byte
      \param src1 - 8bit image upper byte
      \param src2 - 8bit image lower byte
      \param dest - 16bit image
      \return true is succeeded. false otherwise
      */
    bool merge16BitFrame(Mat &src1,Mat &src2,Mat &dest);
    /*! Write image into file
      *\param ext - file extension (for example ".png")
      *\param img - the image to save (8 bit par chanel)
      */
    inline void writeImage(char* ext,const Mat& img);
    /*! Read the current image
      *\return allocated decompressed image
      */
    inline Mat readImage();

    uint32_t lenght; //!< The length of the file
    uint32_t curpos; //!< Current position
    int fps; //!< Frames par second
    mode curmod; //!< Current saving mode
    std::ofstream *outfile; //!< File output stream
    std::ifstream *infile; //!< File input stream
    std::vector<uint32_t> positions;

    uint16_t t_gamma[2048];//!< Gamma correction (not used now)
};

#endif // KINNECTSTORAGE_H
