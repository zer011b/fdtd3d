#include <iostream>
#include <fstream>
#include <sstream>

#include "BMPDumper.h"

void
BMPDumper::dumpGrid (Grid& grid)
{
#if defined (GRID_1D)
  dump1D (grid);
#else
#if defined (GRID_2D)
  dump2D (grid);
#else
#if defined (GRID_3D)
  dump3D (grid);
#endif
#endif
#endif
}

RGBApixel
BMPDumper::setPixel (const FieldValue& val, const FieldValue& maxNeg,
                     const FieldValue& max)
{
  RGBApixel pixel;
  pixel.Alpha = 1.0;

  double value = val - maxNeg;
  if (value > max / 2.0)
  {
    value -= max / 2;
    FieldValue tmp = 2 * value / max;
    pixel.Red = tmp * 255;
    pixel.Green = (1.0 - tmp) * 255;
    pixel.Blue = 0.0;
  }
  else
  {  
    FieldValue tmp = 0;
    if (max == 0)
    {
      tmp = 0.0;
    }
    else
    {
      tmp = 2 * value / max;
    }

    pixel.Red = 0.0;
    pixel.Green = tmp * 255;
    pixel.Blue = (1.0 - tmp) * 255;
  }

  return pixel;
}

#if defined (GRID_1D)
void
BMPDumper::dump1D (Grid& grid)
{
  GridCoordinate& size = grid.getSize ();
  grid_coord sx = size.getX ();

  std::cout << "Saving 1D to BMP image. Size: " << sx << "x1. " << std::endl;

  // Create image for current values and max/min values
  BMP imageCur;
  imageCur.SetSize (sx, 1);
  imageCur.SetBitDepth (24);

  FieldValue maxPosCur = grid.getValues ()[0].getCurValue ();
  FieldValue maxNegCur = grid.getValues ()[0].getCurValue ();

  // Create image for previous values and max/min values
#if defined (ONE_TIME_STEP) || defined (TWO_TIME_STEPS)
  BMP imagePrev;
  imagePrev.SetSize (sx, 1);
  imagePrev.SetBitDepth (24);

  FieldValue maxPosPrev = grid.getValues ()[0].getPrevValue ();
  FieldValue maxNegPrev = grid.getValues ()[0].getPrevValue ();

  // Create image for previous previous values and max/min values
#if defined (TWO_TIME_STEPS)
  BMP imagePrevPrev;
  imagePrevPrev.SetSize (sx, 1);
  imagePrevPrev.SetBitDepth (24);

  FieldValue maxPosPrevPrev = grid.getValues ()[0].getPrevPrevValue ();
  FieldValue maxNegPrevPrev = grid.getValues ()[0].getPrevPrevValue ();
#endif
#endif

  // Calculate max values
  for (FieldPointValue& current : grid.getValues ())
  {
    FieldValue& cur = current.getCurValue ();
    if (cur > maxPosCur)
    {
      maxPosCur = cur;
    }
    if (cur < maxNegCur)
    {
      maxNegCur = cur;
    }

#if defined (ONE_TIME_STEP) || defined (TWO_TIME_STEPS)
    FieldValue& prev = current.getPrevValue ();
    if (prev > maxPosPrev)
    {
      maxPosPrev = prev;
    }
    if (prev < maxNegPrev)
    {
      maxNegPrev = prev;
    }

#if defined (TWO_TIME_STEPS)
    FieldValue& prevPrev = current.getPrevPrevValue ();
    if (prevPrev > maxPosPrevPrev)
    {
      maxPosPrevPrev = prevPrev;
    }
    if (prevPrev < maxNegPrevPrev)
    {
      maxNegPrevPrev = prevPrev;
    }
#endif
#endif
  }

  FieldValue maxCur = maxPosCur - maxNegCur;
#if defined (ONE_TIME_STEP) || defined (TWO_TIME_STEPS)
  FieldValue maxPrev = maxPosPrev - maxNegPrev;
#if defined (TWO_TIME_STEPS)
  FieldValue maxPrevPrev = maxPosPrevPrev - maxNegPrevPrev;
#endif
#endif

  VectorFieldPointValues& values = grid.getValues ();
  grid_iter end = values.size ();
  for (grid_iter iter = 0; iter < end; ++iter)
  {
    FieldPointValue& current = values[iter];

    FieldValue& cur = current.getCurValue ();
    RGBApixel pixelCur = setPixel (cur, maxNegCur, maxCur);
#if defined (ONE_TIME_STEP) || defined (TWO_TIME_STEPS)
    FieldValue& prev = current.getPrevValue ();
    RGBApixel pixelPrev = setPixel (prev, maxNegPrev, maxPrev);
#if defined (TWO_TIME_STEPS)
    FieldValue& prevPrev = current.getPrevPrevValue ();
    RGBApixel pixelPrevPrev = setPixel (prevPrev, maxNegPrevPrev, maxPrevPrev);
#endif
#endif

    imageCur.SetPixel(iter, 0, pixelCur);
#if defined (ONE_TIME_STEP) || defined (TWO_TIME_STEPS)
    imagePrev.SetPixel(iter, 0, pixelPrev);
#if defined (TWO_TIME_STEPS)
    imagePrevPrev.SetPixel(iter, 0, pixelPrevPrev);
#endif
#endif
  }

  std::stringstream cur;
  cur << "cur.bmp";
  imageCur.WriteToFile(cur.str().c_str());
#if defined (ONE_TIME_STEP) || defined (TWO_TIME_STEPS)
  std::stringstream prev;
  prev << "prev.bmp";
  imagePrev.WriteToFile(prev.str().c_str());
#if defined (TWO_TIME_STEPS)
  std::stringstream prevPrev;
  prevPrev << "prevPrev.bmp";
  imagePrevPrev.WriteToFile(prevPrev.str().c_str());
#endif
#endif

  std::cout << "Saved. " << std::endl;
}
#endif

#if defined (GRID_2D)
void
BMPDumper::dump2D (Grid& grid)
{
/*  BMP image;

  image.SetSize(sizeX, sizeY);
  image.SetBitDepth(24);

  std::cout << "Saving to BMP image. " << sizeX << " " << sizeY << std::endl;

	double maxP = data[0][0][0];
	double maxM = data[0][0][0];
	for (int i = 0; i < sizeX; ++i)
	for (int j = 0; j < sizeY; ++j)
  //for (int k = 0; k < sizeZ; ++k)
  {
  	if (data[i][j][K] > maxP)
    	maxP = data[i][j][K];
  	if (data[i][j][K] < maxM)
    	maxM = data[i][j][K];
  }
	double max = maxP - maxM;

	for (int i = 0; i < image.TellWidth(); ++i)
	for (int j = 0; j < image.TellHeight(); ++j)
  {
  	RGBApixel a;
  	a.Alpha = 1.0;

//    if (data[i][j] > 0)
//    {
//  	a.Red = 0.0;// data[i][j] * 255 / max;
//  	a.Blue = 0.0;// data[i][j] * 255 / max;
//  	a.Green = data[i][j] * 255 / max;
//    }
//  	else
//    {
//  	a.Red = -data[i][j] * 255 / max;
//  	a.Blue = 0.0;// data[i][j] * 255 / max;
//  	a.Green = 0.0;// data[i][j] * 255 / max;
//    }

  	double value = data[i][j][K] - maxM;
  	if (value > max / 2.0)
    {
    	value -= max / 2;
    	float tmp = 2 * value / max;
    	a.Red = tmp * 255;
    	a.Green = (1.0 - tmp) * 255;
    	a.Blue = 0.0;

      //std::cout << "!" << tmp * 255 << " " << (1.0 - tmp) * 255 << " " << maxP << " " << maxM << " " <<
      //	max << " " << maxP - maxM << std::endl;
    }
  	else
    {  
    	double tmp;
    	if (max == 0)
      	tmp = 0.0;
    	else
      	tmp = 2 * value / max;
    	a.Red = 0.0;
    	a.Green = tmp * 255;
    	a.Blue = (1.0 - tmp) * 255;
    }

  	image.SetPixel(i, j, a);
  }

  //std::string fff("mkdir ");
  //fff += dest;
  //system(fff.c_str());

	std::stringstream s;
	s << dest << "\\" << filename;

	image.WriteToFile(s.str().c_str());

	std::cout << "Saved. " << std::endl;*/
}
#endif

#if defined (GRID_3D)
void
BMPDumper::dump3D (Grid& grid)
{

}
#endif