
#define _USE_MATH_DEFINES
#include <stdlib.h>
#include <iostream>
#include <fstream>
#include <sstream>
#include <cmath>
#include <mpi.h>
#include <omp.h>

#include "EasyBMP.h"

#define DEBUG 0

//===Physics=========================
#define PI 3.1415926535897932384626433832795
#define CC 2.99792458e+8 //скорость света в вакууме
#define CCC 2.99792458e+10 //скорость света в вакууме
#define MU_Z 0.0000012566370614359173//постоянная магнитной пр-ти в вакууме  4.0*M_PI*1.0e-7 
#define EPS_Z 0.0000000000088541878176203892 //диэлектрическая проницаемость 1.0/(CC*CC*MU_Z)
#define FREQ (CC/LAMBDA) // частота источника
#define LAMBDA (0.000003) // чтоб лишние разы не делить при фикс частоте можно забить вручную
#define OMEGA (2.0*M_PI*FREQ) // та же история
#define IMP_Z sqrt(MU_Z/EPS_Z)
#define PERIOD (1/FREQ)

//===================================
double allocatedCount;
double deallocatedCount;

//===Fields==========================
double*** Ex;
double*** Ey;
double*** Ez;

double*** Dx;
double*** Dx_prev;
double*** Dy;
double*** Dy_prev;
double*** Dz;
double*** Dz_prev;

double*** CaDx;
double*** CbDx;
double*** CaDy;
double*** CbDy;
double*** CaDz;
double*** CbDz;

double** CaDx_join;
double** CbDx_join;
double** Dx_join_right;
double** Dx_join_right_prev;
double** CaDy_join;
double** CbDy_join;
double** Dy_join_right;
double** Dy_join_right_prev;

double** DaBx_join;
double** DbBx_join;
double** Bx_join_right;
double** Bx_join_right_prev;
double** DaBy_join;
double** DbBy_join;
double** By_join_right;
double** By_join_right_prev;

double** CaEz_join;
double** CbEz_join;
double** CcEz_join;
double** Ez_join_right;
double** DaHz_join;
double** DbHz_join;
double** DcHz_join;
double** Hz_join_right;

double*** Hx;
double*** Hy;
double*** Hz;

double*** Bx;
double*** Bx_prev;
double*** By;
double*** By_prev;
double*** Bz;
double*** Bz_prev;

double*** DaBx;
double*** DbBx;
double*** DaBy;
double*** DbBy;
double*** DaBz;
double*** DbBz;

double*** Eps;
double*** Mu;

double*** Sigma;
double*** Sigma1;

double*** CaEx;
double*** CbEx;
double*** CcEx;
double*** CaEy;
double*** CbEy;
double*** CcEy;
double*** CaEz;
double*** CbEz;
double*** CcEz;

double*** DaHx;
double*** DbHx;
double*** DcHx;
double*** DaHy;
double*** DbHy;
double*** DcHy;
double*** DaHz;
double*** DbHz;
double*** DcHz;

int gridX;
int gridY;
int gridZ;

//===UPML============================
int pmlL, pmlR;

double boundaryFactor;
double exponent;

//===TFSF============================
/*int tfL, tfR;

double *E_inc, *H_inc;
int inc_size;*/

//===All data========================
double*** All_Ex;
double** All_Ex2D;
double* buf3D;
double* rbuf3D;

double*** All_Ez;
double** All_Ez2D;

//===================================
MPI_Status status;

int rank, numProcs;

int step;
int numTimeSteps;
int amplitudeSteps;

double dx, dy, dz, dt;
double delta;

double** Hx_buf;
double** Hy_buf;

double** Ex_buf;
double** Ey_buf;

double** Dx_buf;
double** Dy_buf;
double** Dz_buf;
double** Dx_buf_prev;
double** Dy_buf_prev;
double** Dz_buf_prev;

double** Bx_buf;
double** By_buf;
double** Bz_buf;
double** Bx_buf_prev;
double** By_buf_prev;
double** Bz_buf_prev;

double* bufx;
double* bufy;
double* bufz;
double* bufk;

double* rbufx;
double* rbufy;
double* rbufz;
double* rbufk;

int sliceSize, addSize;

MPI_Request requestE1;
MPI_Request requestE2;

MPI_Request requestH1;
MPI_Request requestH2;

MPI_Request requestDx1;
MPI_Request requestDx1_prev;
MPI_Request requestDy1;
MPI_Request requestDy1_prev;
MPI_Request requestDz1;
MPI_Request requestDz1_prev;

MPI_Request requestBx1;
MPI_Request requestBx1_prev;
MPI_Request requestBy1;
MPI_Request requestBy1_prev;
MPI_Request requestBz1;
MPI_Request requestBz1_prev;

//==Meta================================
double*** omegaMP;
double*** omegaEP;
double*** gammaM;
double*** gammaE;

double*** Dx1;
double*** Dx1_prev;
double*** Dy1;
double*** Dy1_prev;
double*** Dz1;
double*** Dz1_prev;

double*** Dx_prev2;
double*** Dy_prev2;
double*** Dz_prev2;

double*** Bx1;
double*** Bx1_prev;
double*** By1;
double*** By1_prev;
double*** Bz1;
double*** Bz1_prev;

double*** Bx_prev2;
double*** By_prev2;
double*** Bz_prev2;

double*** bm0_x;
double*** bm1_x;
double*** bm2_x;
double*** am1_x;
double*** am2_x;
double*** A_x;

double*** bm0_y;
double*** bm1_y;
double*** bm2_y;
double*** am1_y;
double*** am2_y;
double*** A_y;

double*** bm0_z;
double*** bm1_z;
double*** bm2_z;
double*** am1_z;
double*** am2_z;
double*** A_z;

double*** dm0_x;
double*** dm1_x;
double*** dm2_x;
double*** cm1_x;
double*** cm2_x;
double*** C_x;

double*** dm0_y;
double*** dm1_y;
double*** dm2_y;
double*** cm1_y;
double*** cm2_y;
double*** C_y;

double*** dm0_z;
double*** dm1_z;
double*** dm2_z;
double*** cm1_z;
double*** cm2_z;
double*** C_z;

double** Dx1_join_right;
double** Dx1_join_right_prev;
double** Dy1_join_right;
double** Dy1_join_right_prev;
double** Bx1_join_right;
double** Bx1_join_right_prev;
double** By1_join_right;
double** By1_join_right_prev;

double** Dx_join_right_prev2;
double** Dy_join_right_prev2;
double** Bx_join_right_prev2;
double** By_join_right_prev2;

double** bm0_x_join_right;
double** bm1_x_join_right;
double** bm2_x_join_right;
double** am1_x_join_right;
double** am2_x_join_right;
double** A_x_join_right;

double** bm0_y_join_right;
double** bm1_y_join_right;
double** bm2_y_join_right;
double** am1_y_join_right;
double** am2_y_join_right;
double** A_y_join_right;

double** dm0_x_join_right;
double** dm1_x_join_right;
double** dm2_x_join_right;
double** cm1_x_join_right;
double** cm2_x_join_right;
double** C_x_join_right;

double** dm0_y_join_right;
double** dm1_y_join_right;
double** dm2_y_join_right;
double** cm1_y_join_right;
double** cm2_y_join_right;
double** C_y_join_right;

double*** Dx1_prev2;
double*** Dy1_prev2;
double*** Dz1_prev2;
double*** Bx1_prev2;
double*** By1_prev2;
double*** Bz1_prev2;

double** Dx1_join_right_prev2;
double** Dy1_join_right_prev2;
double** Bx1_join_right_prev2;
double** By1_join_right_prev2;

//==Precisions==========================
double acc0 = 0.000000000000001;

//===Amplitude==========================
bool calculateAmplitude;
double*** Ex_amp;
double*** Ey_amp;
double*** Ez_amp;
double*** Hx_amp;
double*** Hy_amp;
double*** Hz_amp;

double** Ex_join_right_amp;
double** Ey_join_right_amp;
double** Hz_join_right_amp;

double* timeAmp;

double W = 0.0;
double *W1;
double *Wstep;
double p0;

//===File Saving==========================================================
void saveToBMP(double* data, int sizeX, char* dest, const char* filename)
{
	

	std::cout << "Saving to BMP image. " << std::endl;

	double maxP = data[0];
	double maxM = data[0];
	for (int i = 0; i < sizeX; ++i)
	{
		if (data[i] > maxP)
			maxP = data[i];
		if (data[i] < maxM)
			maxM = data[i];
	}
	double max = maxP > std::fabs(maxM) ? maxP : std::fabs(maxM);
	//std::cout << "MAX = " << max << " " << maxP << " " << maxM << std::endl;
	BMP image;
	image.SetSize(sizeX, 255 + 5);
	image.SetBitDepth(24);

	for (int i = 0; i < image.TellWidth(); ++i)
	{
		for (int j = 0; j < image.TellHeight(); ++j)
		{
			RGBApixel a;
			a.Alpha = 1.0;
			a.Red = 0.0;
			a.Green = 0.0;
			a.Blue = 0.0;

			image.SetPixel(i, j, a);
		}
	}

	for (int i = 0; i < image.TellWidth(); ++i)
	{
		for (int j = 0; j < 255.0*std::fabs(data[i])/max; ++j)
		{
			//std::cout << "++" << i << " " << j << " " << 255.0*std::fabs(data[i]) / max << std::endl;
			RGBApixel a;
			a.Alpha = 1.0;
			if (i < numTimeSteps)
			{
				a.Red = 255.0;
				a.Green = 255;
				a.Blue = 255.0;
			}
			else 
			{
				a.Red = 255;
				a.Green = 0.0;
				a.Blue = 0.0;
			}

			image.SetPixel(i, j, a);
		}
	}

	//std::string fff("mkdir ");
	//fff += dest;
	//system(fff.c_str());

	std::stringstream s;
	s << dest << "\\" << filename;

	image.WriteToFile(s.str().c_str());

	std::cout << "Saved. " << std::endl;
}
void saveToBMP(double** data, int sizeX, int sizeY, char* dest, const char* filename)
{
	BMP image;
	image.SetSize(sizeX, sizeY);
	image.SetBitDepth(24);

	std::cout << "Saving to BMP image. " << std::endl;

	double maxP = data[0][0];
	double maxM = data[0][0];
	for (int i = 0; i < sizeX; ++i)
	for (int j = 0; j < sizeY; ++j)
	{
		if (data[i][j] > maxP)
			maxP = data[i][j];
		if (data[i][j] < maxM)
			maxM = data[i][j];
	}
	double max = maxP - maxM;

	for (int i = 0; i < image.TellWidth(); ++i)
	for (int j = 0; j < image.TellHeight(); ++j)
	{
		RGBApixel a;
		a.Alpha = 1.0;

		/*if (data[i][j] > 0)
		{
			a.Red = 0.0;// data[i][j] * 255 / max;
			a.Blue = 0.0;// data[i][j] * 255 / max;
			a.Green = data[i][j] * 255 / max;
		}
		else
		{
			a.Red = -data[i][j] * 255 / max;
			a.Blue = 0.0;// data[i][j] * 255 / max;
			a.Green = 0.0;// data[i][j] * 255 / max;
		}*/

		double value = data[i][j] - maxM;
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
			double tmp = 2 * value / max;
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

	std::cout << "Saved. " << std::endl;
}
void saveToBMP(double*** data, int sizeX, int sizeY, int sizeZ, char* dest, const char* filename, int K)
{
	BMP image;
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

		/*if (data[i][j] > 0)
		{
		a.Red = 0.0;// data[i][j] * 255 / max;
		a.Blue = 0.0;// data[i][j] * 255 / max;
		a.Green = data[i][j] * 255 / max;
		}
		else
		{
		a.Red = -data[i][j] * 255 / max;
		a.Blue = 0.0;// data[i][j] * 255 / max;
		a.Green = 0.0;// data[i][j] * 255 / max;
		}*/

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

	std::cout << "Saved. " << std::endl;
}

void writeToFile(double *data, int size, char* dest, const char* filename)
{
	std::stringstream s;
	s << dest << "\\" << filename;

	std::ofstream file(s.str().c_str(), std::ios::out);
	double tmp;

	if (!file.is_open())
	{
		std::cout << "ERROR::Cannot open file for write!";
		return;
	}

	int lll = 0;

	for (int i = 0; i < size; i++)
	{
			//tmp = Eps[i][j];
			tmp = data[i];
			//if (abs(tmp) < acc0)
			//	tmp = 0.0;
			//if (tmp != 0)	
			file << tmp;
		file << std::endl;
		++lll;
	}
	//cout << "Length = " << lll << endl;
	//cout << dataTM->gridX*dataTM->gridY*sizeof(double) << " BYTM written succesfully" << endl;

	file.close();
}
void writeToFile(double **data, int sizeX, int sizeY, char* dest, const char* filename)
{
	//std::string fff("mkdir ");
	//fff += dest;
	//system(fff.c_str());

	std::stringstream s;
	s << dest << "\\" << filename;

	std::ofstream file(s.str().c_str(), std::ios::out);
	double tmp;

	if (!file.is_open())
	{
		std::cout << "ERROR::Cannot open file for write!";
		return;
	}

	int lll = 0;

	for (int i = 0; i < sizeX; i++){
		for (int j = 0; j < sizeY; j++)
		{
			//tmp = Eps[i][j];
			tmp = data[i][j];
			//if (abs(tmp) < acc0)
			//	tmp = 0.0;
			//if (tmp != 0)	
			file << tmp;
			file << " ";
		}
		file << std::endl;
		++lll;
	}
	//cout << "Length = " << lll << endl;
	//cout << dataTM->gridX*dataTM->gridY*sizeof(double) << " BYTM written succesfully" << endl;

	file.close();
}
void writeToFile(double ***data, int sizeX, int sizeY, int sizeZ, char* dest, const char* filename, int K)
{
	//std::string fff("mkdir ");
	//fff += dest;
	//system(fff.c_str());

	std::stringstream s;
	s << dest << "\\" << filename;

	std::ofstream file(s.str().c_str(), std::ios::out);
	double tmp;

	if (!file.is_open())
	{
		std::cout << "ERROR::Cannot open file for write!";
		return;
	}

	int lll = 0;

	for (int i = 0; i < sizeX; i++)
	{
		for (int j = 0; j < sizeY; j++)
		{
			//for (int k = 0; k < sizeZ; ++k)
			//{
				//tmp = Eps[i][j];
				tmp = data[i][j][K];
				if (abs(tmp) < acc0)
					tmp = 0.0;
				//if (tmp != 0)	
				file << tmp;
				file << " ";
				//std::cout << j << " " << k << std::endl;
			//}
		}
		file << std::endl;
		++lll;
	}
	//cout << "Length = " << lll << endl;
	//cout << dataTM->gridX*dataTM->gridY*sizeof(double) << " BYTM written succesfully" << endl;

	file.close();
}

//===Memory Allocation====================================================
double*** Malloc3darray(int x, int y, int z)
{
	double*** data;
	int i, j, k;
	data = new double**[x];
	for (i = 0; i < x; i++)
	{
		data[i] = new double*[y];
		for (j = 0; j < y; j++)
		{
			data[i][j] = new double[z];
			for (k = 0; k < z; ++k)
				data[i][j][k] = 0.0;
		}
	}
	allocatedCount += x*y*z;
	return data;
}
double** Malloc2darray(int x, int y)
{
	double** data;
	int i, j;
	data = new double*[x];
	for (i = 0; i < x; i++)
	{
		data[i] = new double[y];
		for (j = 0; j < y; ++j)
			data[i][j] = 0.0;
	}
	allocatedCount += x*y;
	return data;
}

void Dealloc3darray(double*** data, int x, int y, int z)
{
	int i, j;
	for (i = 0; i < x; i++)
	{
		for (j = 0; j < y; j++)
		{
			delete[] data[i][j];
			//if (rank == 1)
			//	std::cout << i << " of " << x << " + " << j << " of " << y << std::endl;
		}
		delete[] data[i];
	}
	delete[] data;
	deallocatedCount += x * y * z;
}
void Dealloc2darray(double** data, int x, int y)
{
	int i;
	for (i = 0; i < x; i++)
	{
		delete[] data[i];
	}
	delete[] data;
	deallocatedCount += x * y;
}

void Copy3darray(double*** to, double*** from, int x, int y, int z)
{
	int i, j, k;
	for (i = 0; i < x; i++)
	{
		for (j = 0; j < y; j++)
		{
			for (k = 0; k < z; ++k)
				to[i][j][k] = from[i][j][k];
		}
	}
}
void Copy2darray(double** to, double** from, int x, int y)
{
	int i, j;
	for (i = 0; i < x; i++)
	{
		for (j = 0; j < y; j++)
			to[i][j] = from[i][j];
	}
}
//===Saving full field file==============================================
/*void gatherAllEx()
{
	for (int r = 1; r < numProcs; ++r)
	{
		//Ex gather
		if (r == rank)
		{
			for (int k = 0; k <= sliceSize - 2; ++k)
			for (int i = 0; i <= gridX - 1; i++)
			for (int j = 0; j <= gridY - 2; j++)
				buf3D[k*(gridY - 1)*gridX + i*(gridY - 1) + j] = Ex[i][j][k];
				
			MPI_Send(buf3D, gridX*(gridY - 1)*(sliceSize - 1), MPI_DOUBLE, 0, rank, MPI_COMM_WORLD);
		}
		else if (rank == 0)
		{
			MPI_Recv(rbuf3D, gridX*(gridY - 1)*(sliceSize - 1), MPI_DOUBLE, r, r, MPI_COMM_WORLD, &status);

			for (int k = 0; k <= sliceSize - 2; ++k)
			for (int i = 0; i <= gridX - 1; i++)
			for (int j = 0; j <= gridY - 2; j++)
				All_Ex[i][j][k + r * sliceSize] = rbuf3D[k*(gridY - 1)*gridX + i*(gridY - 1) + j];
			
		}
		MPI_Barrier(MPI_COMM_WORLD);
		
		//Ex join gather
		if (r == rank && r != numProcs - 1)
		{
			for (int i = 0; i <= gridX - 1; i++)
			for (int j = 0; j <= gridY - 2; j++)
				buf3D[i*(gridY - 1) + j] = Ex_join_right[i][j];

			MPI_Send(buf3D, gridX*(gridY - 1), MPI_DOUBLE, 0, rank, MPI_COMM_WORLD);
		}
		else if (rank == 0 && r != numProcs - 1)
		{
			MPI_Recv(rbuf3D, gridX*(gridY - 1), MPI_DOUBLE, r, r, MPI_COMM_WORLD, &status);

			for (int i = 0; i <= gridX - 1; i++)
			for (int j = 0; j <= gridY - 2; j++)
				All_Ex[i][j][(r + 1) * sliceSize - 1] = rbuf3D[i*(gridY - 1) + j];
		}
		MPI_Barrier(MPI_COMM_WORLD);
	}

	if (rank == 0)
	{
		for (int k = 0; k <= sliceSize - 2; ++k)
		for (int i = 0; i <= gridX - 1; i++)
		for (int j = 0; j <= gridY - 2; j++)
			All_Ex[i][j][k] = Ex[i][j][k];

		for (int i = 0; i <= gridX - 1; i++)
		for (int j = 0; j <= gridY - 2; j++)
			All_Ex[i][j][sliceSize - 1] = Ex_join_right[i][j];
	}
}

void AllEx()
{
	if (rank == 0)
		All_Ex = Malloc3darray(gridX, gridY - 1, gridZ - 1);
	if (rank != 0)
		buf3D = new double[gridX*(gridY - 1)*(sliceSize - 1)];
	else
		rbuf3D = new double[gridX*(gridY - 1)*(sliceSize - 1)];

	gatherAllEx();

	if (rank == 0)
	{
		std::string buf("All_Ex[");
		std::string buf1("All_Ex[");
		std::ostringstream oss;
		oss << step;
		buf += oss.str();
		buf1 += oss.str();
		buf.append("].bmp");
		buf1.append("].txt");
		writeToFile(All_Ex[gridX/2], gridY - 1, gridZ - 1, "D:\\fdtd", buf1.c_str());
		//saveToBMP(All_Ex, gridX, gridY - 1, gridZ - 1, "D:\\fdtd", buf.c_str(), gridY - 3);
		//saveToBMP(All_Ex, gridX, gridY - 1, gridZ - 1, "D:\\fdtd", buf1.c_str(), 1);
		saveToBMP(All_Ex[gridX / 2], gridY - 1, gridZ - 1, "D:\\fdtd", buf.c_str());
	}

	if (rank == 0)
		Dealloc3darray(All_Ex, gridX, gridY - 1, gridZ - 1);

	if (rank != 0)
		delete[] buf3D;
	else
		delete[] rbuf3D;
}


void gatherAllEx2D()
{
	for (int r = 1; r < numProcs; ++r)
	{
		//Ex gather
		if (r == rank)
		{
			for (int k = 0; k <= sliceSize - 2; ++k)
			//for (int i = 0; i <= gridX - 1; i++)
			for (int j = 0; j <= gridY - 2; j++)
				buf3D[k*(gridY - 1) + j] = Ex[gridX / 2][j][k];

			MPI_Send(buf3D, (gridY - 1)*(sliceSize - 1), MPI_DOUBLE, 0, rank, MPI_COMM_WORLD);
		}
		else if (rank == 0)
		{
			MPI_Recv(rbuf3D, (gridY - 1)*(sliceSize - 1), MPI_DOUBLE, r, r, MPI_COMM_WORLD, &status);

			for (int k = 0; k <= sliceSize - 2; ++k)
			//for (int i = 0; i <= gridX - 1; i++)
			for (int j = 0; j <= gridY - 2; j++)
				All_Ex2D[j][k + r * sliceSize] = rbuf3D[k*(gridY - 1) + j];

		}
		MPI_Barrier(MPI_COMM_WORLD);

		//Ex join gather
		if (r == rank && r != numProcs - 1)
		{
			//for (int i = 0; i <= gridX - 1; i++)
			for (int j = 0; j <= gridY - 2; j++)
				buf3D[j] = Ex_join_right[gridX/2][j];

			MPI_Send(buf3D, (gridY - 1), MPI_DOUBLE, 0, rank, MPI_COMM_WORLD);
		}
		else if (rank == 0 && r != numProcs - 1)
		{
			MPI_Recv(rbuf3D, (gridY - 1), MPI_DOUBLE, r, r, MPI_COMM_WORLD, &status);

			//for (int i = 0; i <= gridX - 1; i++)
			for (int j = 0; j <= gridY - 2; j++)
				All_Ex2D[j][(r + 1) * sliceSize - 1] = rbuf3D[j];
		}
		MPI_Barrier(MPI_COMM_WORLD);
	}

	if (rank == 0)
	{
		for (int k = 0; k <= sliceSize - 2; ++k)
		//for (int i = 0; i <= gridX - 1; i++)
		for (int j = 0; j <= gridY - 2; j++)
			All_Ex2D[j][k] = Ex[gridX/2][j][k];

		//for (int i = 0; i <= gridX - 1; i++)
		for (int j = 0; j <= gridY - 2; j++)
			All_Ex2D[j][sliceSize - 1] = Ex_join_right[gridX/2][j];
	}
}
void AllEx2D()
{
	if (rank == 0)
		All_Ex2D = Malloc2darray(gridY - 1, gridZ - 1);
	if (rank != 0)
		buf3D = new double[(gridY - 1)*(sliceSize - 1)];
	else
		rbuf3D = new double[(gridY - 1)*(sliceSize - 1)];

	gatherAllEx2D();

	if (rank == 0)
	{
		std::string buf("All_Ex[");
		std::string buf1("All_Ex[");
		std::ostringstream oss;
		oss << step;
		buf += oss.str();
		buf1 += oss.str();
		buf.append("].bmp");
		buf1.append("].txt");
		writeToFile(All_Ex2D, gridY - 1, gridZ - 1, "D:\\fdtd", buf1.c_str());
		//saveToBMP(All_Ex, gridX, gridY - 1, gridZ - 1, "D:\\fdtd", buf.c_str(), gridY - 3);
		//saveToBMP(All_Ex, gridX, gridY - 1, gridZ - 1, "D:\\fdtd", buf1.c_str(), 1);
		saveToBMP(All_Ex2D, gridY - 1, gridZ - 1, "D:\\fdtd", buf.c_str());
	}

	if (rank == 0)
		Dealloc2darray(All_Ex2D, gridY - 1, gridZ - 1);

	if (rank != 0)
		delete[] buf3D;
	else
		delete[] rbuf3D;
}


void gatherAllEz2D()
{
	for (int r = 1; r < numProcs; ++r)
	{
		//Ez gather
		if (r == rank)
		{
			for (int k = 0; k <= sliceSize - 1; ++k)
				//for (int i = 0; i <= gridX - 1; i++)
			for (int j = 0; j <= gridY - 2; j++)
				buf3D[k*(gridY - 1) + j] = Ez[gridX / 2][j][k];

			MPI_Send(buf3D, (gridY - 1)*(sliceSize), MPI_DOUBLE, 0, rank, MPI_COMM_WORLD);
		}
		else if (rank == 0)
		{
			MPI_Recv(rbuf3D, (gridY - 1)*(sliceSize), MPI_DOUBLE, r, r, MPI_COMM_WORLD, &status);

			for (int k = 0; k <= sliceSize - 1; ++k)
				//for (int i = 0; i <= gridX - 1; i++)
			for (int j = 0; j <= gridY - 2; j++)
				All_Ez2D[j][k + r * sliceSize] = rbuf3D[k*(gridY - 1) + j];

		}
		MPI_Barrier(MPI_COMM_WORLD);

	}

	if (rank == 0)
	{
		for (int k = 0; k <= sliceSize - 1; ++k)
			//for (int i = 0; i <= gridX - 1; i++)
		for (int j = 0; j <= gridY - 2; j++)
			All_Ez2D[j][k] = Ez[gridX / 2][j][k];

		//for (int i = 0; i <= gridX - 1; i++)
		//for (int j = 0; j <= gridY - 2; j++)
		//	All_Ex2D[j][sliceSize - 1] = Ex_join_right[gridX / 2][j];
	}
}
void AllEz2D()
{
	if (rank == 0)
		All_Ez2D = Malloc2darray(gridY - 1, gridZ);
	if (rank != 0)
		buf3D = new double[(gridY - 1)*(sliceSize)];
	else
		rbuf3D = new double[(gridY - 1)*(sliceSize)];

	gatherAllEz2D();

	if (rank == 0)
	{
		std::string buf("All_Ez[");
		std::string buf1("All_Ez[");
		std::ostringstream oss;
		oss << step;
		buf += oss.str();
		buf1 += oss.str();
		buf.append("].bmp");
		buf1.append("].txt");
		writeToFile(All_Ez2D, gridY - 1, gridZ, "D:\\fdtd", buf1.c_str());
		//saveToBMP(All_Ex, gridX, gridY - 1, gridZ - 1, "D:\\fdtd", buf.c_str(), gridY - 3);
		//saveToBMP(All_Ex, gridX, gridY - 1, gridZ - 1, "D:\\fdtd", buf1.c_str(), 1);
		saveToBMP(All_Ez2D, gridY - 1, gridZ, "D:\\fdtd", buf.c_str());
	}

	if (rank == 0)
		Dealloc2darray(All_Ez2D, gridY - 1, gridZ);

	if (rank != 0)
		delete[] buf3D;
	else
		delete[] rbuf3D;
}
*/

//======================Update FDTD===============================
void UpdateE()
{
	#pragma omp parallel for
	for (int i = 0; i <= gridX - 2; i++)
	for (int j = 1; j <= gridY - 2; j++)
	for (int k = 1; k <= sliceSize - 2; k++)
	{
		//if (i == gridX / 2 && j == gridY / 2 && k == gridZ*1.0 / 4 - 1 && rank == 0)
		//	std::cout << "Ex[i][j][k] == " << Ex[i][j][k] << " = " << CaEx[i][j][k] << " * " << Ex[i][j][k] << " + " <<
		//	CbEx[i][j][k] << " * " << Dx1[i][j][k] << " - " << CcEx[i][j][k] << " * " << Dx1_prev[i][j][k] << std::endl;

		double tmpD1 = (Dx1[i][j - 1][k - 1] + Dx1[i + 1][j - 1][k - 1] + 
			Dx1[i][j][k - 1] + Dx1[i + 1][j][k - 1] + 
			Dx1[i][j - 1][k] + Dx1[i + 1][j - 1][k] +
			Dx1[i][j][k] + Dx1[i + 1][j][k]) / 8;
		double tmpD1_prev = (Dx1_prev[i][j - 1][k - 1] + Dx1_prev[i + 1][j - 1][k - 1] +
			Dx1_prev[i][j][k - 1] + Dx1_prev[i + 1][j][k - 1] +
			Dx1_prev[i][j - 1][k] + Dx1_prev[i + 1][j - 1][k] +
			Dx1_prev[i][j][k] + Dx1_prev[i + 1][j][k]) / 8;

		Ex[i][j][k] = CaEx[i][j][k] * Ex[i][j][k] +
					  CbEx[i][j][k] * tmpD1 - 
					  CcEx[i][j][k] * tmpD1_prev;
	}

	#pragma omp parallel for
	for (int i = 1; i <= gridX - 2; i++)
	for (int j = 0; j <= gridY - 2; j++)
	for (int k = 1; k <= sliceSize - 2; k++)
	{
		double tmpD1 = (Dy1[i - 1][j][k - 1] + Dy1[i - 1][j + 1][k - 1] +
			Dy1[i][j][k - 1] + Dy1[i][j + 1][k - 1] + 
			Dy1[i - 1][j][k] + Dy1[i - 1][j + 1][k] +
			Dy1[i][j][k] + Dy1[i][j + 1][k]) / 8;
		double tmpD1_prev = (Dy1_prev[i - 1][j][k - 1] + Dy1_prev[i - 1][j + 1][k - 1] +
			Dy1_prev[i][j][k - 1] + Dy1_prev[i][j + 1][k - 1] +
			Dy1_prev[i - 1][j][k] + Dy1_prev[i - 1][j + 1][k] +
			Dy1_prev[i][j][k] + Dy1_prev[i][j + 1][k]) / 8;

		Ey[i][j][k] = CaEy[i][j][k] * Ey[i][j][k] +
					  CbEy[i][j][k] * tmpD1 -
					  CcEy[i][j][k] * tmpD1_prev;
	}

	#pragma omp parallel for
	for (int i = 1; i <= gridX - 2; i++)
	for (int j = 1; j <= gridY - 2; j++)
	for (int k = 0; k <= sliceSize - 2; k++)
	{
		//if(i == gridX / 2 && j == gridY / 2 && k == gridZ*1.0 / 4 - 1 && rank == 0)
		//	std::cout << "Ez[i][j][k] == " << Ez[i][j][k] << " = " << CaEz[i][j][k] << " * " << Ez[i][j][k] << " + " <<
		//	CbEz[i][j][k] << " * " << Dz1[i][j][k] << " - " << CcEz[i][j][k] << " * " << Dz1_prev[i][j][k] << std::endl;
		
		double tmpD1 = (Dz1[i - 1][j - 1][k] + Dz1[i - 1][j - 1][k + 1] +
			Dz1[i][j - 1][k] + Dz1[i][j - 1][k + 1] + 
			Dz1[i - 1][j][k] + Dz1[i - 1][j][k + 1] + 
			Dz1[i][j][k] + Dz1[i][j][k + 1]) / 8;
		double tmpD1_prev = (Dz1_prev[i - 1][j - 1][k] + Dz1_prev[i - 1][j - 1][k + 1] +
			Dz1_prev[i][j - 1][k] + Dz1_prev[i][j - 1][k + 1] +
			Dz1_prev[i - 1][j][k] + Dz1_prev[i - 1][j][k + 1] +
			Dz1_prev[i][j][k] + Dz1_prev[i][j][k + 1]) / 8;

		Ez[i][j][k] = CaEz[i][j][k] * Ez[i][j][k] +
					  CbEz[i][j][k] * tmpD1 -
					  CcEz[i][j][k] * tmpD1_prev;

		//if (i == gridX / 2 && j == gridY / 2 && k == sliceSize / 2 && rank == 0)
		//	std::cout << "tmpD1 = " << tmpD1 << std::endl;
	}
}
void UpdateD()
{
	#pragma omp parallel for
	for (int i = 0; i <= gridX - 1; i++)
	for (int j = 0; j <= gridY - 2; j++)
	for (int k = 0; k <= sliceSize - 2; k++)
	{
		//if (i == gridX / 2 && j == gridY / 2 && k == gridZ*1.0 / 4-1 && rank == 0)
		//	std::cout << "====================" << std::endl;
		//if (i == gridX / 2 && j == gridY / 2 && k == gridZ*1.0/4-1 && rank == 0)
		//	std::cout << "Dx[i][j][k] == " << Dx[i][j][k] << " = "<< CaDx[i][j][k] << " * " << Dx_prev[i][j][k] << " + " <<
		//	CbDx[i][j][k] << " * (" << Hz[i][j + 1][k] << " - " << Hz[i][j][k] << " + " <<
		//	Hy[i][j][k] << " - " << Hy[i][j][k + 1] << ") / " << delta << std::endl;
		Dx[i][j][k] = CaDx[i][j][k] * Dx_prev[i][j][k] + CbDx[i][j][k] *
			((Hz[i][j + 1][k] - Hz[i][j][k])/delta + (Hy[i][j][k] - Hy[i][j][k + 1]) / delta);
	}

	#pragma omp parallel for
	for (int i = 0; i <= gridX - 2; i++)
	for (int j = 0; j <= gridY - 1; j++)
	for (int k = 0; k <= sliceSize - 2; k++)
	{
		Dy[i][j][k] = CaDy[i][j][k] * Dy_prev[i][j][k] + CbDy[i][j][k] *
			((Hx[i][j][k + 1] - Hx[i][j][k])/delta + (Hz[i][j][k] - Hz[i + 1][j][k]) / delta);
	}

	#pragma omp parallel for
	for (int i = 0; i <= gridX - 2; i++)
	for (int j = 0; j <= gridY - 2; j++)
	for (int k = 0; k <= sliceSize - 1; k++)
	{
		//if (i == gridX / 2 && j == gridY / 2 && k == gridZ*1.0 / 4 - 1 && rank == 0)
		//	std::cout << "Dz[i][j][k] == " << Dz[i][j][k] << " = " << CaDz[i][j][k] << " * " << Dz_prev[i][j][k] << " + " <<
		//	CbDz[i][j][k] << " * (" << Hy[i+1][j][k] << " - " << Hy[i][j][k] << " + " <<
		//	Hx[i][j][k] << " - " << Hx[i][j+1][k] << ") / " << delta << std::endl;

		Dz[i][j][k] = CaDz[i][j][k] * Dz_prev[i][j][k] + CbDz[i][j][k] *
			((Hy[i + 1][j][k] - Hy[i][j][k]) / delta + (Hx[i][j][k] - Hx[i][j + 1][k]) / delta);
	}

	//#pragma omp parallel for
	for (int k = 0; k <= sliceSize - 1; ++k)
	{
		Dz[gridX / 2][gridY / 2][k] -= - cos(OMEGA*dt*step) / (2 * 10000000000);
	}
	//if (rank == 0)
	//	Dz[gridX / 2][gridY / 2][sliceSize/2] -= -cos(OMEGA*dt*step) / (2 * 10000000000);
}
void UpdateE_shared_1()
{
	if (rank != numProcs - 1)
	{
		//Ez in join
		#pragma omp parallel for
		for (int i = 1; i <= gridX - 2; i++)
		for (int j = 1; j <= gridY - 2; j++)
		{
			double tmpD1 = (Dz1[i - 1][j - 1][sliceSize - 1] + Dz_buf[i - 1][j - 1] +
				Dz1[i][j - 1][sliceSize - 1] + Dz_buf[i][j - 1] +
				Dz1[i - 1][j][sliceSize - 1] + Dz_buf[i - 1][j] +
				Dz1[i][j][sliceSize - 1] + Dz_buf[i][j]) / 8;
			double tmpD1_prev = (Dz1_prev[i - 1][j - 1][sliceSize - 1] + Dz_buf_prev[i - 1][j - 1] +
				Dz1_prev[i][j - 1][sliceSize - 1] + Dz_buf_prev[i][j - 1] +
				Dz1_prev[i - 1][j][sliceSize - 1] + Dz_buf_prev[i - 1][j] +
				Dz1_prev[i][j][sliceSize - 1] + Dz_buf_prev[i][j]) / 8;

			Ez_join_right[i][j] = CaEz_join[i][j] * Ez_join_right[i][j] +
				CbEz_join[i][j] * tmpD1 -
				CcEz_join[i][j] * tmpD1_prev;
		}
	}
}
void UpdateE_shared_2()
{
	if (rank != 0)
	{
		//Ex[][][0]
		#pragma omp parallel for
		for (int i = 0; i <= gridX - 2; i++)
		for (int j = 1; j <= gridY - 2; j++)
		{
			double tmpD1 = (Dx_buf[i][j - 1] + Dx_buf[i + 1][j - 1] +
				Dx_buf[i][j] + Dx_buf[i + 1][j] +
				Dx1[i][j - 1][0] + Dx1[i + 1][j - 1][0] +
				Dx1[i][j][0] + Dx1[i + 1][j][0]) / 8;
			double tmpD1_prev = (Dx_buf_prev[i][j - 1] + Dx_buf_prev[i + 1][j - 1] +
				Dx_buf_prev[i][j] + Dx_buf_prev[i + 1][j] +
				Dx1_prev[i][j - 1][0] + Dx1_prev[i + 1][j - 1][0] +
				Dx1_prev[i][j][0] + Dx1_prev[i + 1][j][0]) / 8;

			Ex[i][j][0] = CaEx[i][j][0] * Ex[i][j][0] +
				CbEx[i][j][0] * tmpD1 -
				CcEx[i][j][0] * tmpD1_prev;
		}

		//Ey[][][0]
		#pragma omp parallel for
		for (int i = 1; i <= gridX - 2; i++)
		for (int j = 0; j <= gridY - 2; j++)
		{
			double tmpD1 = (Dy_buf[i - 1][j] + Dy_buf[i - 1][j + 1] +
				Dy_buf[i][j] + Dy_buf[i][j + 1] +
				Dy1[i - 1][j][0] + Dy1[i - 1][j + 1][0] +
				Dy1[i][j][0] + Dy1[i][j + 1][0]) / 8;
			double tmpD1_prev = (Dy_buf_prev[i - 1][j] + Dy_buf_prev[i - 1][j + 1] +
				Dy_buf_prev[i][j] + Dy_buf_prev[i][j + 1] +
				Dy1_prev[i - 1][j][0] + Dy1_prev[i - 1][j + 1][0] +
				Dy1_prev[i][j][0] + Dy1_prev[i][j + 1][0]) / 8;

			Ey[i][j][0] = CaEy[i][j][0] * Ey[i][j][0] +
				CbEy[i][j][0] * tmpD1 -
				CcEy[i][j][0] * tmpD1_prev;
		}
	}
}
void UpdateD_shared()
{
	if (rank != numProcs - 1)
	{
		#pragma omp parallel for
		for (int i = 0; i <= gridX - 1; i++)
		for (int j = 0; j <= gridY - 2; j++)
		{
			Dx_join_right[i][j] = CaDx_join[i][j] * Dx_join_right_prev[i][j] + CbDx_join[i][j] *
				((Hz_join_right[i][j + 1] - Hz_join_right[i][j]) / delta + (Hy[i][j][sliceSize - 1] - Hy_buf[i][j]) / delta);
		}

		#pragma omp parallel for
		for (int i = 0; i <= gridX - 2; i++)
		for (int j = 0; j <= gridY - 1; j++)
		{
			Dy_join_right[i][j] = CaDy_join[i][j] * Dy_join_right_prev[i][j] + CbDy_join[i][j] *
				((Hx_buf[i][j] - Hx[i][j][sliceSize - 1]) / delta + (Hz_join_right[i][j] - Hz_join_right[i + 1][j]) / delta);
		}
	}
}
void ResetE()
{
	#pragma omp parallel for
	for (int i = 0; i <= gridX - 1; i++)
	for (int j = 0; j <= gridY - 2; j++)
	for (int k = 0; k <= sliceSize - 2; k++)
	{
		Dx_prev2[i][j][k] = Dx_prev[i][j][k];
		Dx_prev[i][j][k] = Dx[i][j][k];
		Dx1_prev2[i][j][k] = Dx1_prev[i][j][k];
		Dx1_prev[i][j][k] = Dx1[i][j][k];
	}

	#pragma omp parallel for
	for (int i = 0; i <= gridX - 2; i++)
	for (int j = 0; j <= gridY - 1; j++)
	for (int k = 0; k <= sliceSize - 2; k++)
	{
		Dy_prev2[i][j][k] = Dy_prev[i][j][k];
		Dy_prev[i][j][k] = Dy[i][j][k];
		Dy1_prev2[i][j][k] = Dy1_prev[i][j][k];
		Dy1_prev[i][j][k] = Dy1[i][j][k];
	}

	#pragma omp parallel for
	for (int i = 0; i <= gridX - 2; i++)
	for (int j = 0; j <= gridY - 2; j++)
	for (int k = 0; k <= sliceSize - 1; k++)
	{
		Dz_prev2[i][j][k] = Dz_prev[i][j][k];
		Dz_prev[i][j][k] = Dz[i][j][k];
		Dz1_prev2[i][j][k] = Dz1_prev[i][j][k];
		Dz1_prev[i][j][k] = Dz1[i][j][k];
	}

	if (rank != numProcs - 1)
	{
		#pragma omp parallel for
		for (int i = 0; i <= gridX - 1; i++)
		for (int j = 0; j <= gridY - 2; j++)
		{
			Dx_join_right_prev2[i][j] = Dx_join_right_prev[i][j];
			Dx_join_right_prev[i][j] = Dx_join_right[i][j];
			Dx1_join_right_prev2[i][j] = Dx1_join_right_prev[i][j];
			Dx1_join_right_prev[i][j] = Dx1_join_right[i][j];
		}

		#pragma omp parallel for
		for (int i = 0; i <= gridX - 2; i++)
		for (int j = 0; j <= gridY - 1; j++)
		{
			Dy_join_right_prev2[i][j] = Dy_join_right_prev[i][j];
			Dy_join_right_prev[i][j] = Dy_join_right[i][j];
			Dy1_join_right_prev2[i][j] = Dy1_join_right_prev[i][j];
			Dy1_join_right_prev[i][j] = Dy1_join_right[i][j];
		}
	}
}

void UpdateD1()
{
	#pragma omp parallel for
	for (int i = 0; i <= gridX - 1; i++)
	for (int j = 0; j <= gridY - 2; j++)
	for (int k = 0; k <= sliceSize - 2; k++)
	{
		//if (i == gridX / 2 && j == gridY / 2 && k == gridZ*1.0 / 4 - 1 && rank == 0)
		//	std::cout << "Dx1[i][j][k] == " << Dx1[i][j][k] << " = (" << bm0_x[i][j][k] << " * " << Dx[i][j][k] << " + " <<
		//	bm1_x[i][j][k] << " * " << Dx_prev[i][j][k] << " + " << bm2_x[i][j][k] << " * " <<
		//	Dx_prev2[i][j][k] << " - " << am1_x[i][j][k] << " * " << Dx1_prev[i][j][k] << " - " <<
		//	am2_x[i][j][k] << " * " << Dx1_prev2[i][j][k] << ") / " << A_x[i][j][k] << std::endl;
		
		Dx1[i][j][k] = (bm0_x[i][j][k] * Dx[i][j][k] + bm1_x[i][j][k] * Dx_prev[i][j][k] +
			bm2_x[i][j][k] * Dx_prev2[i][j][k] - am1_x[i][j][k] * Dx1_prev[i][j][k] -
			am2_x[i][j][k] * Dx1_prev2[i][j][k]) / A_x[i][j][k];
	}

	#pragma omp parallel for
	for (int i = 0; i <= gridX - 2; i++)
	for (int j = 0; j <= gridY - 1; j++)
	for (int k = 0; k <= sliceSize - 2; k++)
	{
		Dy1[i][j][k] = (bm0_y[i][j][k] * Dy[i][j][k] + bm1_y[i][j][k] * Dy_prev[i][j][k] +
			bm2_y[i][j][k] * Dy_prev2[i][j][k] - am1_y[i][j][k] * Dy1_prev[i][j][k] -
			am2_y[i][j][k] * Dy1_prev2[i][j][k]) / A_y[i][j][k];
	}

	#pragma omp parallel for
	for (int i = 0; i <= gridX - 2; i++)
	for (int j = 0; j <= gridY - 2; j++)
	for (int k = 0; k <= sliceSize - 1; k++)
	{
		Dz1[i][j][k] = (bm0_z[i][j][k] * Dz[i][j][k] + bm1_z[i][j][k] * Dz_prev[i][j][k] +
			bm2_z[i][j][k] * Dz_prev2[i][j][k] - am1_z[i][j][k] * Dz1_prev[i][j][k] -
			am2_z[i][j][k] * Dz1_prev2[i][j][k]) / A_z[i][j][k];
	}
}
void UpdateD1_shared()
{
	if (rank != numProcs - 1)
	{
		#pragma omp parallel for
		for (int i = 0; i <= gridX - 1; i++)
		for (int j = 0; j <= gridY - 2; j++)
		{
			Dx1_join_right[i][j] = (bm0_x_join_right[i][j] * Dx_join_right[i][j] +
				bm1_x_join_right[i][j] * Dx_join_right_prev[i][j] +
				bm2_x_join_right[i][j] * Dx_join_right_prev2[i][j] -
				am1_x_join_right[i][j] * Dx1_join_right_prev[i][j] -
				am2_x_join_right[i][j] * Dx1_join_right_prev2[i][j]) / A_x_join_right[i][j];
		}

		#pragma omp parallel for
		for (int i = 0; i <= gridX - 2; i++)
		for (int j = 0; j <= gridY - 1; j++)
		{
			Dy1_join_right[i][j] = (bm0_y_join_right[i][j] * Dy_join_right[i][j] +
				bm1_y_join_right[i][j] * Dy_join_right_prev[i][j] +
				bm2_y_join_right[i][j] * Dy_join_right_prev2[i][j] -
				am1_y_join_right[i][j] * Dy1_join_right_prev[i][j] -
				am2_y_join_right[i][j] * Dy1_join_right_prev2[i][j]) / A_y_join_right[i][j];
		}
	}
}

void UpdateH()
{
	#pragma omp parallel for
	for (int i = 0; i <= gridX - 2; i++)
	for (int j = 1; j <= gridY - 2; j++)
	for (int k = 1; k <= sliceSize - 2; k++)
	{
		//if (i == gridX / 2 && j == gridY / 2 && k == gridZ*1.0 / 4 - 1 && rank == 0)
		//	std::cout << "Ex[i][j][k] == " << Ex[i][j][k] << " = " << CaEx[i][j][k] << " * " << Ex[i][j][k] << " + " <<
		//	CbEx[i][j][k] << " * " << Dx1[i][j][k] << " - " << CcEx[i][j][k] << " * " << Dx1_prev[i][j][k] << std::endl;

		double tmpB1 = (Bx1[i][j - 1][k - 1] + Bx1[i + 1][j - 1][k - 1] +
			Bx1[i][j][k - 1] + Bx1[i + 1][j][k - 1] +
			Bx1[i][j - 1][k] + Bx1[i + 1][j - 1][k] +
			Bx1[i][j][k] + Bx1[i + 1][j][k]) / 8;
		double tmpB1_prev = (Bx1_prev[i][j - 1][k - 1] + Bx1_prev[i + 1][j - 1][k - 1] +
			Bx1_prev[i][j][k - 1] + Bx1_prev[i + 1][j][k - 1] +
			Bx1_prev[i][j - 1][k] + Bx1_prev[i + 1][j - 1][k] +
			Bx1_prev[i][j][k] + Bx1_prev[i + 1][j][k]) / 8;

		Hx[i][j][k] = DaHx[i][j][k] * Hx[i][j][k] +
			DbHx[i][j][k] * tmpB1 -
			DcHx[i][j][k] * tmpB1_prev;
	}

	#pragma omp parallel for
	for (int i = 1; i <= gridX - 2; i++)
	for (int j = 0; j <= gridY - 2; j++)
	for (int k = 1; k <= sliceSize - 2; k++)
	{
		double tmpB1 = (By1[i - 1][j][k - 1] + By1[i - 1][j + 1][k - 1] +
			By1[i][j][k - 1] + By1[i][j + 1][k - 1] +
			By1[i - 1][j][k] + By1[i - 1][j + 1][k] +
			By1[i][j][k] + By1[i][j + 1][k]) / 8;
		double tmpB1_prev = (By1_prev[i - 1][j][k - 1] + By1_prev[i - 1][j + 1][k - 1] +
			By1_prev[i][j][k - 1] + By1_prev[i][j + 1][k - 1] +
			By1_prev[i - 1][j][k] + By1_prev[i - 1][j + 1][k] +
			By1_prev[i][j][k] + By1_prev[i][j + 1][k]) / 8;

		Hy[i][j][k] = DaHy[i][j][k] * Hy[i][j][k] +
			DbHy[i][j][k] * tmpB1 -
			DcHy[i][j][k] * tmpB1_prev;
	}

	#pragma omp parallel for
	for (int i = 1; i <= gridX - 2; i++)
	for (int j = 1; j <= gridY - 2; j++)
	for (int k = 0; k <= sliceSize - 2; k++)
	{
		double tmpB1 = (Bz1[i - 1][j - 1][k] + Bz1[i - 1][j - 1][k + 1] +
			Bz1[i][j - 1][k] + Bz1[i][j - 1][k + 1] +
			Bz1[i - 1][j][k] + Bz1[i - 1][j][k + 1] +
			Bz1[i][j][k] + Bz1[i][j][k + 1]) / 8;
		double tmpB1_prev = (Bz1_prev[i - 1][j - 1][k] + Bz1_prev[i - 1][j - 1][k + 1] +
			Bz1_prev[i][j - 1][k] + Bz1_prev[i][j - 1][k + 1] +
			Bz1_prev[i - 1][j][k] + Bz1_prev[i - 1][j][k + 1] +
			Bz1_prev[i][j][k] + Bz1_prev[i][j][k + 1]) / 8;

		Hz[i][j][k] = DaHz[i][j][k] * Hz[i][j][k] +
			DbHz[i][j][k] * tmpB1 -
			DcHz[i][j][k] * tmpB1_prev;
	}
}
void UpdateB()
{
	#pragma omp parallel for
	for (int i = 0; i <= gridX - 1; i++)
	for (int j = 0; j <= gridY - 2; j++)
	for (int k = 0; k <= sliceSize - 2; k++)
	{
		Bx[i][j][k] = DaBx[i][j][k] * Bx_prev[i][j][k] + DbBx[i][j][k] *
			((Ey[i][j][k + 1] - Ey[i][j][k]) / delta + (Ez[i][j][k] - Ez[i][j + 1][k]) / delta);
	}

	#pragma omp parallel for
	for (int i = 0; i <= gridX - 2; i++)
	for (int j = 0; j <= gridY - 1; j++)
	for (int k = 0; k <= sliceSize - 2; k++)
	{
		By[i][j][k] = DaBy[i][j][k] * By_prev[i][j][k] + DbBy[i][j][k] *
			((Ez[i + 1][j][k] - Ez[i][j][k]) / delta + (Ex[i][j][k] - Ex[i][j][k + 1]) / delta);
	}

	#pragma omp parallel for
	for (int i = 0; i <= gridX - 2; i++)
	for (int j = 0; j <= gridY - 2; j++)
	for (int k = 0; k <= sliceSize - 1; k++)
	{
		Bz[i][j][k] = DaBz[i][j][k] * Bz_prev[i][j][k] + DbBz[i][j][k] *
			((Ex[i][j + 1][k] - Ex[i][j][k]) / delta + (Ey[i][j][k] - Ey[i + 1][j][k]) / delta);
	}
}
void UpdateH_shared_1()
{
	if (rank != numProcs - 1)
	{
		//Hz in join
		#pragma omp parallel for
		for (int i = 1; i <= gridX - 2; i++)
		for (int j = 1; j <= gridY - 2; j++)
		{
			double tmpB1 = (Bz1[i - 1][j - 1][sliceSize - 1] + Bz_buf[i - 1][j - 1] +
				Bz1[i][j - 1][sliceSize - 1] + Bz_buf[i][j - 1] +
				Bz1[i - 1][j][sliceSize - 1] + Bz_buf[i - 1][j] +
				Bz1[i][j][sliceSize - 1] + Bz_buf[i][j]) / 8;
			double tmpB1_prev = (Bz1_prev[i - 1][j - 1][sliceSize - 1] + Bz_buf_prev[i - 1][j - 1] +
				Bz1_prev[i][j - 1][sliceSize - 1] + Bz_buf_prev[i][j - 1] +
				Bz1_prev[i - 1][j][sliceSize - 1] + Bz_buf_prev[i - 1][j] +
				Bz1_prev[i][j][sliceSize - 1] + Bz_buf_prev[i][j]) / 8;

			Hz_join_right[i][j] = DaHz_join[i][j] * Hz_join_right[i][j] +
				DbHz_join[i][j] * tmpB1 -
				DcHz_join[i][j] * tmpB1_prev;
		}
	}
}
void UpdateH_shared_2()
{
	if (rank != 0)
	{
		//Hx[][][0]
		#pragma omp parallel for
		for (int i = 0; i <= gridX - 2; i++)
		for (int j = 1; j <= gridY - 2; j++)
		{
			double tmpB1 = (Bx_buf[i][j - 1] + Bx_buf[i + 1][j - 1] +
				Bx_buf[i][j] + Bx_buf[i + 1][j] +
				Bx1[i][j - 1][0] + Bx1[i + 1][j - 1][0] +
				Bx1[i][j][0] + Bx1[i + 1][j][0]) / 8;
			double tmpB1_prev = (Bx_buf_prev[i][j - 1] + Bx_buf_prev[i + 1][j - 1] +
				Bx_buf_prev[i][j] + Bx_buf_prev[i + 1][j] +
				Bx1_prev[i][j - 1][0] + Bx1_prev[i + 1][j - 1][0] +
				Bx1_prev[i][j][0] + Bx1_prev[i + 1][j][0]) / 8;

			Hx[i][j][0] = DaHx[i][j][0] * Hx[i][j][0] +
				DbHx[i][j][0] * tmpB1 -
				DcHx[i][j][0] * tmpB1_prev;
		}

		//Hy[][][0]
		#pragma omp parallel for
		for (int i = 1; i <= gridX - 2; i++)
		for (int j = 0; j <= gridY - 2; j++)
		{
			double tmpB1 = (By_buf[i - 1][j] + By_buf[i - 1][j + 1] +
				By_buf[i][j] + By_buf[i][j + 1] +
				By1[i - 1][j][0] + By1[i - 1][j + 1][0] +
				By1[i][j][0] + By1[i][j + 1][0]) / 8;
			double tmpB1_prev = (By_buf_prev[i - 1][j] + By_buf_prev[i - 1][j + 1] +
				By_buf_prev[i][j] + By_buf_prev[i][j + 1] +
				By1_prev[i - 1][j][0] + By1_prev[i - 1][j + 1][0] +
				By1_prev[i][j][0] + By1_prev[i][j + 1][0]) / 8;

			Hy[i][j][0] = DaHy[i][j][0] * Hy[i][j][0] +
				DbHy[i][j][0] * tmpB1 -
				DcHy[i][j][0] * tmpB1_prev;
		}
	}
}
void UpdateB_shared()
{
	if (rank != numProcs - 1)
	{
		#pragma omp parallel for
		for (int i = 0; i <= gridX - 1; i++)
		for (int j = 0; j <= gridY - 2; j++)
		{
			Bx_join_right[i][j] = DaBx_join[i][j] * Bx_join_right_prev[i][j] + DbBx_join[i][j] *
				((Ey_buf[i][j] - Ey[i][j][sliceSize - 1]) / delta + (Ez_join_right[i][j] - Ez_join_right[i][j + 1]) / delta);
		}

		#pragma omp parallel for
		for (int i = 0; i <= gridX - 2; i++)
		for (int j = 0; j <= gridY - 1; j++)
		{
			By_join_right[i][j] = DaBy_join[i][j] * By_join_right_prev[i][j] + DbBy_join[i][j] *
				((Ez_join_right[i + 1][j] - Ez_join_right[i][j]) / delta + (Ex[i][j][sliceSize - 1] - Ex_buf[i][j]) / delta);
		}
	}
}
void ResetH()
{
	#pragma omp parallel for
	for (int i = 0; i <= gridX - 1; i++)
	for (int j = 0; j <= gridY - 2; j++)
	for (int k = 0; k <= sliceSize - 2; k++)
	{
		Bx_prev2[i][j][k] = Bx_prev[i][j][k];
		Bx_prev[i][j][k] = Bx[i][j][k];
		Bx1_prev2[i][j][k] = Bx1_prev[i][j][k];
		Bx1_prev[i][j][k] = Bx1[i][j][k];
	}

	#pragma omp parallel for
	for (int i = 0; i <= gridX - 2; i++)
	for (int j = 0; j <= gridY - 1; j++)
	for (int k = 0; k <= sliceSize - 2; k++)
	{
		By_prev2[i][j][k] = By_prev[i][j][k];
		By_prev[i][j][k] = By[i][j][k];
		By1_prev2[i][j][k] = By1_prev[i][j][k];
		By1_prev[i][j][k] = By1[i][j][k];
	}

	#pragma omp parallel for
	for (int i = 0; i <= gridX - 2; i++)
	for (int j = 0; j <= gridY - 2; j++)
	for (int k = 0; k <= sliceSize - 1; k++)
	{
		Bz_prev2[i][j][k] = Bz_prev[i][j][k];
		Bz_prev[i][j][k] = Bz[i][j][k];
		Bz1_prev2[i][j][k] = Bz1_prev[i][j][k];
		Bz1_prev[i][j][k] = Bz1[i][j][k];
	}

	if (rank != numProcs - 1)
	{
	#pragma omp parallel for
		for (int i = 0; i <= gridX - 1; i++)
		for (int j = 0; j <= gridY - 2; j++)
		{
			Bx_join_right_prev2[i][j] = Bx_join_right_prev[i][j];
			Bx_join_right_prev[i][j] = Bx_join_right[i][j];
			Bx1_join_right_prev2[i][j] = Bx1_join_right_prev[i][j];
			Bx1_join_right_prev[i][j] = Bx1_join_right[i][j];
		}

	#pragma omp parallel for
		for (int i = 0; i <= gridX - 2; i++)
		for (int j = 0; j <= gridY - 1; j++)
		{
			By_join_right_prev2[i][j] = By_join_right_prev[i][j];
			By_join_right_prev[i][j] = By_join_right[i][j];
			By1_join_right_prev2[i][j] = By1_join_right_prev[i][j];
			By1_join_right_prev[i][j] = By1_join_right[i][j];
		}
	}
}

void UpdateB1()
{
	#pragma omp parallel for
	for (int i = 0; i <= gridX - 1; i++)
	for (int j = 0; j <= gridY - 2; j++)
	for (int k = 0; k <= sliceSize - 2; k++)
	{
		//if (i == gridX / 2 && j == gridY / 2 && k == gridZ*1.0 / 4 - 1 && rank == 0)
		//	std::cout << "Dx1[i][j][k] == " << Dx1[i][j][k] << " = (" << bm0_x[i][j][k] << " * " << Dx[i][j][k] << " + " <<
		//	bm1_x[i][j][k] << " * " << Dx_prev[i][j][k] << " + " << bm2_x[i][j][k] << " * " <<
		//	Dx_prev2[i][j][k] << " - " << am1_x[i][j][k] << " * " << Dx1_prev[i][j][k] << " - " <<
		//	am2_x[i][j][k] << " * " << Dx1_prev2[i][j][k] << ") / " << A_x[i][j][k] << std::endl;

		Bx1[i][j][k] = (dm0_x[i][j][k] * Bx[i][j][k] + dm1_x[i][j][k] * Bx_prev[i][j][k] +
			dm2_x[i][j][k] * Bx_prev2[i][j][k] - cm1_x[i][j][k] * Bx1_prev[i][j][k] -
			cm2_x[i][j][k] * Bx1_prev2[i][j][k]) / C_x[i][j][k];
	}

	#pragma omp parallel for
	for (int i = 0; i <= gridX - 2; i++)
	for (int j = 0; j <= gridY - 1; j++)
	for (int k = 0; k <= sliceSize - 2; k++)
	{
		By1[i][j][k] = (dm0_y[i][j][k] * By[i][j][k] + dm1_y[i][j][k] * By_prev[i][j][k] +
			dm2_y[i][j][k] * By_prev2[i][j][k] - cm1_y[i][j][k] * By1_prev[i][j][k] -
			cm2_y[i][j][k] * By1_prev2[i][j][k]) / C_y[i][j][k];
	}

	#pragma omp parallel for
	for (int i = 0; i <= gridX - 2; i++)
	for (int j = 0; j <= gridY - 2; j++)
	for (int k = 0; k <= sliceSize - 1; k++)
	{
		Bz1[i][j][k] = (dm0_z[i][j][k] * Bz[i][j][k] + dm1_z[i][j][k] * Bz_prev[i][j][k] +
			dm2_z[i][j][k] * Bz_prev2[i][j][k] - cm1_z[i][j][k] * Bz1_prev[i][j][k] -
			cm2_z[i][j][k] * Bz1_prev2[i][j][k]) / C_z[i][j][k];
	}
}
void UpdateB1_shared()
{
	if (rank != numProcs - 1)
	{
		#pragma omp parallel for
		for (int i = 0; i <= gridX - 1; i++)
		for (int j = 0; j <= gridY - 2; j++)
		{
			Bx1_join_right[i][j] = (dm0_x_join_right[i][j] * Bx_join_right[i][j] +
				dm1_x_join_right[i][j] * Bx_join_right_prev[i][j] +
				dm2_x_join_right[i][j] * Bx_join_right_prev2[i][j] -
				cm1_x_join_right[i][j] * Bx1_join_right_prev[i][j] -
				cm2_x_join_right[i][j] * Bx1_join_right_prev2[i][j]) / C_x_join_right[i][j];
		}

		#pragma omp parallel for
		for (int i = 0; i <= gridX - 2; i++)
		for (int j = 0; j <= gridY - 1; j++)
		{
			By1_join_right[i][j] = (dm0_y_join_right[i][j] * By_join_right[i][j] +
				dm1_y_join_right[i][j] * By_join_right_prev[i][j] +
				dm2_y_join_right[i][j] * By_join_right_prev2[i][j] -
				cm1_y_join_right[i][j] * By1_join_right_prev[i][j] -
				cm2_y_join_right[i][j] * By1_join_right_prev2[i][j]) / C_y_join_right[i][j];
		}
	}
}

//======================TFSF===============================
/*void FixD()
{
    #pragma omp parallel for
	for (int i = tfL; i <= gridX - 1 - tfR; i++)
	for (int j = tfL; j <= gridY - 2 - tfR; j++)
	{
		Dx[i][j][tfL - 1] += CbDx[i][j][tfL - 1] * ((H_inc[tfL] + H_inc[tfL + 1]) / 2) / delta;
		Dx[i][j][gridZ - 2 - tfR + 1] -= CbDx[i][j][gridZ - 2 - tfR + 1] * ((H_inc[inc_size - 2 - tfR] + H_inc[inc_size - 2 - tfR - 1]) / 2) / delta;
	}

    #pragma omp parallel for
	for (int j = tfL; j <= gridY - 2 - tfR; j++)
	for (int k = tfL; k <= sliceSize - 1 - tfR; k++)
	{
		Dz[tfL - 1][j][k] -= CbDz[tfL - 1][j][k] * ((H_inc[tfL] + H_inc[tfL + 1])/2) / delta;
		Dz[gridX - 2 - tfR + 1][j][k] += CbDz[gridX - 2 - tfR + 1][j][k] * ((H_inc[inc_size - 2 - tfR] + H_inc[inc_size - 2 - tfR - 1])/2) / delta;
	}
}

void UpdateIncE()
{
	double velosity = 1;

	#pragma omp parallel for
	for (int i = 1; i < inc_size; i++)
	{
		E_inc[i] += (dt / (velosity * EPS_Z * delta)) * (H_inc[i - 1] - H_inc[i]);
	}
}

void UpdateIncH()
{
	double velocity = 1;

	#pragma omp parallel for
	for (int i = 0; i < inc_size - 1; i++)
	{
		H_inc[i] += (dt / (velocity * MU_Z * delta)) * (E_inc[i] - E_inc[i + 1]);
	}
}
*/
//======================Send===============================
void SendE()
{
	MPI_Request request1;
	MPI_Request request2;

	#pragma omp parallel for
	for (int i = 0; i <= gridX - 2; i++)
	for (int j = 0; j <= gridY - 1; j++)
		bufx[i*gridY + j] = Ex[i][j][0];

	#pragma omp parallel for
	for (int i = 0; i <= gridX - 1; i++)
	for (int j = 0; j <= gridY - 2; j++)
		bufy[i*(gridY - 1) + j] = Ey[i][j][0];

	MPI_Isend(bufx, (gridX - 1)*gridY, MPI_DOUBLE, rank - 1, 2, MPI_COMM_WORLD, &request1);
	MPI_Isend(bufy, gridX*(gridY - 1), MPI_DOUBLE, rank - 1, 3, MPI_COMM_WORLD, &request2);
}
void SendH()
{
	MPI_Request request1;
	MPI_Request request2;

	#pragma omp parallel for
	for (int i = 0; i <= gridX - 2; i++)
	for (int j = 0; j <= gridY - 1; j++)
		bufx[i*gridY + j] = Hx[i][j][0];

	#pragma omp parallel for
	for (int i = 0; i <= gridX - 1; i++)
	for (int j = 0; j <= gridY - 2; j++)
		bufy[i*(gridY - 1) + j] = Hy[i][j][0];

	MPI_Isend(bufx, (gridX - 1)*gridY, MPI_DOUBLE, rank - 1, 2, MPI_COMM_WORLD, &request1);
	MPI_Isend(bufy, gridX*(gridY - 1), MPI_DOUBLE, rank - 1, 3, MPI_COMM_WORLD, &request2);
}
void SendDz()
{
	MPI_Request request1;
	MPI_Request request2;

	#pragma omp parallel for
	for (int i = 0; i <= gridX - 2; i++)
	for (int j = 0; j <= gridY - 2; j++)
		bufx[i*(gridY - 1) + j] = Dz1[i][j][0];

	#pragma omp parallel for
	for (int i = 0; i <= gridX - 2; i++)
	for (int j = 0; j <= gridY - 2; j++)
		bufy[i*(gridY - 1) + j] = Dz1_prev[i][j][0];

	MPI_Isend(bufx, (gridX - 1)*(gridY - 1), MPI_DOUBLE, rank - 1, 4, MPI_COMM_WORLD, &request1);
	MPI_Isend(bufy, (gridX - 1)*(gridY - 1), MPI_DOUBLE, rank - 1, 5, MPI_COMM_WORLD, &request2);
}
void SendDxDy()
{
	MPI_Request request1;
	MPI_Request request2;
	MPI_Request request3;
	MPI_Request request4;

	#pragma omp parallel for
	for (int i = 0; i <= gridX - 1; i++)
	for (int j = 0; j <= gridY - 2; j++)
		bufx[i*(gridY - 1) + j] = Dx1[i][j][0];

	#pragma omp parallel for
	for (int i = 0; i <= gridX - 2; i++)
	for (int j = 0; j <= gridY - 1; j++)
		bufy[i*(gridY) + j] = Dy1[i][j][0];

	#pragma omp parallel for
	for (int i = 0; i <= gridX - 1; i++)
	for (int j = 0; j <= gridY - 2; j++)
		bufz[i*(gridY - 1) + j] = Dx1_prev[i][j][0];

	#pragma omp parallel for
	for (int i = 0; i <= gridX - 2; i++)
	for (int j = 0; j <= gridY - 1; j++)
		bufk[i*(gridY)+j] = Dy1_prev[i][j][0];

	MPI_Isend(bufx, (gridX)*(gridY - 1), MPI_DOUBLE, rank + 1, 6, MPI_COMM_WORLD, &request1);
	MPI_Isend(bufy, (gridX - 1)*(gridY), MPI_DOUBLE, rank + 1, 7, MPI_COMM_WORLD, &request2);
	MPI_Isend(bufz, (gridX)*(gridY - 1), MPI_DOUBLE, rank + 1, 8, MPI_COMM_WORLD, &request1);
	MPI_Isend(bufk, (gridX - 1)*(gridY), MPI_DOUBLE, rank + 1, 9, MPI_COMM_WORLD, &request2);
}
void SendBz()
{
	MPI_Request request1;
	MPI_Request request2;

	#pragma omp parallel for
	for (int i = 0; i <= gridX - 2; i++)
	for (int j = 0; j <= gridY - 2; j++)
		bufx[i*(gridY - 1) + j] = Bz1[i][j][0];

	#pragma omp parallel for
	for (int i = 0; i <= gridX - 2; i++)
	for (int j = 0; j <= gridY - 2; j++)
		bufy[i*(gridY - 1) + j] = Bz1_prev[i][j][0];

	MPI_Isend(bufx, (gridX - 1)*(gridY - 1), MPI_DOUBLE, rank - 1, 10, MPI_COMM_WORLD, &request1);
	MPI_Isend(bufy, (gridX - 1)*(gridY - 1), MPI_DOUBLE, rank - 1, 11, MPI_COMM_WORLD, &request2);
}
void SendBxBy()
{
	MPI_Request request1;
	MPI_Request request2;
	MPI_Request request3;
	MPI_Request request4;

	#pragma omp parallel for
	for (int i = 0; i <= gridX - 1; i++)
	for (int j = 0; j <= gridY - 2; j++)
		bufx[i*(gridY - 1) + j] = Bx1[i][j][0];

	#pragma omp parallel for
	for (int i = 0; i <= gridX - 2; i++)
	for (int j = 0; j <= gridY - 1; j++)
		bufy[i*(gridY)+j] = By1[i][j][0];

	#pragma omp parallel for
	for (int i = 0; i <= gridX - 1; i++)
	for (int j = 0; j <= gridY - 2; j++)
		bufz[i*(gridY - 1) + j] = Bx1_prev[i][j][0];

	#pragma omp parallel for
	for (int i = 0; i <= gridX - 2; i++)
	for (int j = 0; j <= gridY - 1; j++)
		bufk[i*(gridY)+j] = By1_prev[i][j][0];

	MPI_Isend(bufx, (gridX)*(gridY - 1), MPI_DOUBLE, rank + 1, 12, MPI_COMM_WORLD, &request1);
	MPI_Isend(bufy, (gridX - 1)*(gridY), MPI_DOUBLE, rank + 1, 13, MPI_COMM_WORLD, &request2);
	MPI_Isend(bufz, (gridX)*(gridY - 1), MPI_DOUBLE, rank + 1, 14, MPI_COMM_WORLD, &request1);
	MPI_Isend(bufk, (gridX - 1)*(gridY), MPI_DOUBLE, rank + 1, 15, MPI_COMM_WORLD, &request2);
}

void RecvE()
{
	int i, j;

	MPI_Irecv(rbufx, (gridX - 1)*gridY, MPI_DOUBLE, rank + 1, 2, MPI_COMM_WORLD, &requestE1);
	MPI_Irecv(rbufy, gridX*(gridY - 1), MPI_DOUBLE, rank + 1, 3, MPI_COMM_WORLD, &requestE2);

}
void CopyE()
{
	#pragma omp parallel for
	for (int i = 0; i <= gridX - 2; i++)
	for (int j = 0; j <= gridY - 1; j++)
		Ex_buf[i][j] = rbufx[i*gridY + j];

	#pragma omp parallel for
	for (int i = 0; i <= gridX - 1; i++)
	for (int j = 0; j <= gridY - 2; j++)
		Ey_buf[i][j] = rbufy[i*(gridY - 1) + j];
}

void RecvH()
{
	int i, j;

	MPI_Irecv(rbufx, (gridX - 1)*gridY, MPI_DOUBLE, rank + 1, 2, MPI_COMM_WORLD, &requestH1);
	MPI_Irecv(rbufy, gridX*(gridY - 1), MPI_DOUBLE, rank + 1, 3, MPI_COMM_WORLD, &requestH2);
}
void CopyH()
{
	#pragma omp parallel for
	for (int i = 0; i <= gridX - 2; i++)
	for (int j = 0; j <= gridY - 1; j++)
		Hx_buf[i][j] = rbufx[i*gridY + j];

	#pragma omp parallel for
	for (int i = 0; i <= gridX - 1; i++)
	for (int j = 0; j <= gridY - 2; j++)
		Hy_buf[i][j] = rbufy[i*(gridY - 1) + j];
}

void RecvDz()
{
	MPI_Irecv(rbufx, (gridX - 1)*(gridY - 1), MPI_DOUBLE, rank + 1, 4, MPI_COMM_WORLD, &requestDz1);
	MPI_Irecv(rbufy, (gridX - 1)*(gridY - 1), MPI_DOUBLE, rank + 1, 5, MPI_COMM_WORLD, &requestDz1_prev);
}
void CopyDz()
{
	#pragma omp parallel for
	for (int i = 0; i <= gridX - 2; i++)
	for (int j = 0; j <= gridY - 2; j++)
		Dz_buf[i][j] = rbufx[i*(gridY - 1) + j];

	#pragma omp parallel for
	for (int i = 0; i <= gridX - 2; i++)
	for (int j = 0; j <= gridY - 2; j++)
		Dz_buf_prev[i][j] = rbufy[i*(gridY - 1) + j];
}

void RecvBz()
{
	MPI_Irecv(rbufx, (gridX - 1)*(gridY - 1), MPI_DOUBLE, rank + 1, 10, MPI_COMM_WORLD, &requestBz1);
	MPI_Irecv(rbufy, (gridX - 1)*(gridY - 1), MPI_DOUBLE, rank + 1, 11, MPI_COMM_WORLD, &requestBz1_prev);
}
void CopyBz()
{
	#pragma omp parallel for
	for (int i = 0; i <= gridX - 2; i++)
	for (int j = 0; j <= gridY - 2; j++)
		Bz_buf[i][j] = rbufx[i*(gridY - 1) + j];

	#pragma omp parallel for
	for (int i = 0; i <= gridX - 2; i++)
	for (int j = 0; j <= gridY - 2; j++)
		Bz_buf_prev[i][j] = rbufy[i*(gridY - 1) + j];
}

void RecvDxDy()
{
	MPI_Irecv(rbufx, (gridX)*(gridY - 1), MPI_DOUBLE, rank - 1, 6, MPI_COMM_WORLD, &requestDx1);
	MPI_Irecv(rbufy, (gridX - 1)*(gridY), MPI_DOUBLE, rank - 1, 7, MPI_COMM_WORLD, &requestDy1);
	MPI_Irecv(rbufz, (gridX)*(gridY - 1), MPI_DOUBLE, rank - 1, 8, MPI_COMM_WORLD, &requestDx1_prev);
	MPI_Irecv(rbufk, (gridX - 1)*(gridY), MPI_DOUBLE, rank - 1, 9, MPI_COMM_WORLD, &requestDy1_prev);
}
void CopyDxDy()
{
	#pragma omp parallel for
	for (int i = 0; i <= gridX - 1; i++)
	for (int j = 0; j <= gridY - 2; j++)
		Dx_buf[i][j] = rbufx[i*(gridY - 1) + j];

	#pragma omp parallel for
	for (int i = 0; i <= gridX - 2; i++)
	for (int j = 0; j <= gridY - 1; j++)
		Dy_buf[i][j] = rbufy[i*(gridY) + j];

	#pragma omp parallel for
	for (int i = 0; i <= gridX - 1; i++)
	for (int j = 0; j <= gridY - 2; j++)
		Dx_buf_prev[i][j] = rbufz[i*(gridY - 1) + j];

	#pragma omp parallel for
	for (int i = 0; i <= gridX - 2; i++)
	for (int j = 0; j <= gridY - 1; j++)
		Dy_buf_prev[i][j] = rbufk[i*(gridY)+j];
}

void RecvBxBy()
{
	MPI_Irecv(rbufx, (gridX)*(gridY - 1), MPI_DOUBLE, rank - 1, 12, MPI_COMM_WORLD, &requestBx1);
	MPI_Irecv(rbufy, (gridX - 1)*(gridY), MPI_DOUBLE, rank - 1, 13, MPI_COMM_WORLD, &requestBy1);
	MPI_Irecv(rbufz, (gridX)*(gridY - 1), MPI_DOUBLE, rank - 1, 14, MPI_COMM_WORLD, &requestBx1_prev);
	MPI_Irecv(rbufk, (gridX - 1)*(gridY), MPI_DOUBLE, rank - 1, 15, MPI_COMM_WORLD, &requestBy1_prev);
}
void CopyBxBy()
{
	#pragma omp parallel for
	for (int i = 0; i <= gridX - 1; i++)
	for (int j = 0; j <= gridY - 2; j++)
		Bx_buf[i][j] = rbufx[i*(gridY - 1) + j];

	#pragma omp parallel for
	for (int i = 0; i <= gridX - 2; i++)
	for (int j = 0; j <= gridY - 1; j++)
		By_buf[i][j] = rbufy[i*(gridY)+j];

	#pragma omp parallel for
	for (int i = 0; i <= gridX - 1; i++)
	for (int j = 0; j <= gridY - 2; j++)
		Bx_buf_prev[i][j] = rbufz[i*(gridY - 1) + j];

	#pragma omp parallel for
	for (int i = 0; i <= gridX - 2; i++)
	for (int j = 0; j <= gridY - 1; j++)
		By_buf_prev[i][j] = rbufk[i*(gridY)+j];
}
//===UPML==================================================
double Kx(double x)
{
	return 1;
}
double Ky(double y)
{
	return 1;
}
double Kz(double z)
{
	return 1;
}
double SigmaX(double x, double y)
{
	//if (x > maxX)
	//	maxX = x;
	//if (x < minX)
	//	minX = x;
	if (x > pmlL && x < gridX - pmlR + 1)
	{
		//if ((x - gridX / 2 - 1 - 0.5)*(x - gridX / 2 - 1 - 0.5) +
		//	(y - gridY / 2 - 1 - 0.5)*(y - gridY / 2 - 1 - 0.5) < 41)
		//	return OMEGA*0.5 / (4 * PI);

		return 0;
	}
	else
	{
		if (x == pmlL)
		{
			return 0;
		}
		if (x == gridX - pmlR + 1)
		{
			return 0;
		}
		if (x < pmlL)
		{
			double dist = pmlL - x;
			double x1 = (dist + 1) * delta;       // upper bounds for point i
			double x2 = dist * delta;       // lower bounds for point i
			
			return boundaryFactor * (pow(x1, (exponent + 1)) - pow(x2, (exponent + 1)));   //   polynomial grading
		}
		if (x > (gridX - pmlR + 1))
		{
			double dist = x - (gridX - pmlR + 1);
			double x1 = (dist + 1) * delta;       // upper bounds for point i
			double x2 = dist * delta;       // lower bounds for point i

			//std::cout << boundaryFactor * (pow(x1, (exponent + 1)) - pow(x2, (exponent + 1))) << std::endl;
			return boundaryFactor * (pow(x1, (exponent + 1)) - pow(x2, (exponent + 1)));   //   polynomial grading
		}
	}
}
double SigmaY(double x, double y)
{
	//if (y > maxY)
	//	maxY = y;
	//if (y < minY)
	//	minY = y;

	if (y > pmlL && y < gridY - pmlR + 1)
	{
		//if ((x - gridX / 2 - 1 - 0.5)*(x - gridX / 2 - 1 - 0.5) +
		//	(y - gridY / 2 - 1 - 0.5)*(y - gridY / 2 - 1 - 0.5) < 41)
		//	return OMEGA*0.5 / (4 * PI);

		return 0;
	}
	else
	{
		if (y == pmlL)
		{
			return 0;
		}
		if (y == gridY - pmlR + 1)
		{
			return 0;
		}
		if (y < pmlL)
		{
			double dist = pmlL - y;
			double x1 = (dist + 1) * delta;       // upper bounds for point i
			double x2 = dist * delta;       // lower bounds for point i

			return boundaryFactor * (pow(x1, (exponent + 1)) - pow(x2, (exponent + 1)));   //   polynomial grading
		}
		if (y >(gridY - pmlR + 1))
		{
			double dist = y - (gridY - pmlR + 1);
			double x1 = (dist + 1) * delta;       // upper bounds for point i
			double x2 = dist * delta;       // lower bounds for point i

			return boundaryFactor * (pow(x1, (exponent + 1)) - pow(x2, (exponent + 1)));   //   polynomial grading
		}
	}
}
double SigmaZ(double z)
{
	//if (z > maxZ)
	//	maxZ = z;
	//if (z < minZ)
	//	minZ = z;

	return 0;

	if (z > pmlL && z < gridZ - pmlR + 1)
	{
		return 0;
	}
	else
	{
		if (z == pmlL)
		{
			return 0;
		}
		if (z == gridZ - pmlR + 1)
		{
			return 0;
		}
		if (z < pmlL)
		{
			double dist = pmlL - z;
			double x1 = (dist + 1) * delta;       // upper bounds for point i
			double x2 = dist * delta;       // lower bounds for point i

			return boundaryFactor * (pow(x1, (exponent + 1)) - pow(x2, (exponent + 1)));   //   polynomial grading
		}
		if (z > (gridZ - pmlR + 1))
		{
			double dist = z - (gridZ - pmlR + 1);
			double x1 = (dist + 1) * delta;       // upper bounds for point i
			double x2 = dist * delta;       // lower bounds for point i

			return boundaryFactor * (pow(x1, (exponent + 1)) - pow(x2, (exponent + 1)));   //   polynomial grading
		}
	}
}
//=========================================================
void allocateAll()
{
	//E_inc = new double[inc_size];
	//H_inc = new double[inc_size - 1];

	W1 = new double[numTimeSteps + amplitudeSteps];
	Wstep = new double[numTimeSteps + amplitudeSteps];

	//allocate fields======================================
	Ex = Malloc3darray(gridX - 1, gridY, sliceSize);
	Ey = Malloc3darray(gridX, gridY - 1, sliceSize);
	Ez = Malloc3darray(gridX, gridY, sliceSize - 1);

	Hx = Malloc3darray(gridX - 1, gridY, sliceSize);
	Hy = Malloc3darray(gridX, gridY - 1, sliceSize);
	Hz = Malloc3darray(gridX, gridY, sliceSize - 1);

	//allocate D==========================================
	Dx = Malloc3darray(gridX, gridY - 1, sliceSize - 1);
	Dy = Malloc3darray(gridX - 1, gridY, sliceSize - 1);
	Dz = Malloc3darray(gridX - 1, gridY - 1, sliceSize);

	Dx_prev = Malloc3darray(gridX, gridY - 1, sliceSize - 1);
	Dy_prev = Malloc3darray(gridX - 1, gridY, sliceSize - 1);
	Dz_prev = Malloc3darray(gridX - 1, gridY - 1, sliceSize);

	Dx_prev2 = Malloc3darray(gridX, gridY - 1, sliceSize - 1);
	Dy_prev2 = Malloc3darray(gridX - 1, gridY, sliceSize - 1);
	Dz_prev2 = Malloc3darray(gridX - 1, gridY - 1, sliceSize);

	//allocate D1=========================================
	Dx1 = Malloc3darray(gridX, gridY - 1, sliceSize - 1);
	Dy1 = Malloc3darray(gridX - 1, gridY, sliceSize - 1);
	Dz1 = Malloc3darray(gridX - 1, gridY - 1, sliceSize);

	Dx1_prev = Malloc3darray(gridX, gridY - 1, sliceSize - 1);
	Dy1_prev = Malloc3darray(gridX - 1, gridY, sliceSize - 1);
	Dz1_prev = Malloc3darray(gridX - 1, gridY - 1, sliceSize);

	Dx1_prev2 = Malloc3darray(gridX, gridY - 1, sliceSize - 1);
	Dy1_prev2 = Malloc3darray(gridX - 1, gridY, sliceSize - 1);
	Dz1_prev2 = Malloc3darray(gridX - 1, gridY - 1, sliceSize);

	//allocate B=========================================
	Bx = Malloc3darray(gridX, gridY - 1, sliceSize - 1);
	By = Malloc3darray(gridX - 1, gridY, sliceSize - 1);
	Bz = Malloc3darray(gridX - 1, gridY - 1, sliceSize);
	
	Bx_prev = Malloc3darray(gridX, gridY - 1, sliceSize - 1);
	By_prev = Malloc3darray(gridX - 1, gridY, sliceSize - 1);
	Bz_prev = Malloc3darray(gridX - 1, gridY - 1, sliceSize);
	
	Bx_prev2 = Malloc3darray(gridX, gridY - 1, sliceSize - 1);
	By_prev2 = Malloc3darray(gridX - 1, gridY, sliceSize - 1);
	Bz_prev2 = Malloc3darray(gridX - 1, gridY - 1, sliceSize);
	
	//allocate B1========================================
	Bx1 = Malloc3darray(gridX, gridY - 1, sliceSize - 1);
	By1 = Malloc3darray(gridX - 1, gridY, sliceSize - 1);
	Bz1 = Malloc3darray(gridX - 1, gridY - 1, sliceSize);
	
	Bx1_prev = Malloc3darray(gridX, gridY - 1, sliceSize - 1);
	By1_prev = Malloc3darray(gridX - 1, gridY, sliceSize - 1);
	Bz1_prev = Malloc3darray(gridX - 1, gridY - 1, sliceSize);
	
	Bx1_prev2 = Malloc3darray(gridX, gridY - 1, sliceSize - 1);
	By1_prev2 = Malloc3darray(gridX - 1, gridY, sliceSize - 1);
	Bz1_prev2 = Malloc3darray(gridX - 1, gridY - 1, sliceSize);

	//join allocation====================================
	if (rank != numProcs - 1)
	{
		//allocate fields===============================
		Ez_join_right = Malloc2darray(gridX, gridY);
		Hz_join_right = Malloc2darray(gridX, gridY);

		//allocate D and B==============================
		Dx_join_right = Malloc2darray(gridX, gridY - 1);
		Dy_join_right = Malloc2darray(gridX - 1, gridY);
		Bx_join_right = Malloc2darray(gridX, gridY - 1);
		By_join_right = Malloc2darray(gridX - 1, gridY);

		Dx_join_right_prev = Malloc2darray(gridX, gridY - 1);
		Dy_join_right_prev = Malloc2darray(gridX - 1, gridY);
		Bx_join_right_prev = Malloc2darray(gridX, gridY - 1);
		By_join_right_prev = Malloc2darray(gridX - 1, gridY);

		Dx_join_right_prev2 = Malloc2darray(gridX, gridY - 1);
		Dy_join_right_prev2 = Malloc2darray(gridX - 1, gridY);
		Bx_join_right_prev2 = Malloc2darray(gridX, gridY - 1);
		By_join_right_prev2 = Malloc2darray(gridX - 1, gridY);

		//allocate D1 and B1============================
		Dx1_join_right = Malloc2darray(gridX, gridY - 1);
		Dy1_join_right = Malloc2darray(gridX - 1, gridY);
		Bx1_join_right = Malloc2darray(gridX, gridY - 1);
		By1_join_right = Malloc2darray(gridX - 1, gridY);
		
		Dx1_join_right_prev = Malloc2darray(gridX, gridY - 1);
		Dy1_join_right_prev = Malloc2darray(gridX - 1, gridY);
		Bx1_join_right_prev = Malloc2darray(gridX, gridY - 1);
		By1_join_right_prev = Malloc2darray(gridX - 1, gridY);

		Dx1_join_right_prev2 = Malloc2darray(gridX, gridY - 1);
		Dy1_join_right_prev2 = Malloc2darray(gridX - 1, gridY);
		Bx1_join_right_prev2 = Malloc2darray(gridX, gridY - 1);
		By1_join_right_prev2 = Malloc2darray(gridX - 1, gridY);

		//allocate Ca, Cb, Cc, Da, Db, Dc for fields===
		CaEz_join = Malloc2darray(gridX, gridY);
		CbEz_join = Malloc2darray(gridX, gridY);
		CcEz_join = Malloc2darray(gridX, gridY);
		DaHz_join = Malloc2darray(gridX, gridY);
		DbHz_join = Malloc2darray(gridX, gridY);
		DcHz_join = Malloc2darray(gridX, gridY);
		
		//allocate Ca, Cb, Da, Db for D and B==========
		CaDx_join = Malloc2darray(gridX, gridY - 1);
		CbDx_join = Malloc2darray(gridX, gridY - 1);
		CaDy_join = Malloc2darray(gridX - 1, gridY);
		CbDy_join = Malloc2darray(gridX - 1, gridY);
		DaBx_join = Malloc2darray(gridX, gridY - 1);
		DbBx_join = Malloc2darray(gridX, gridY - 1);
		DaBy_join = Malloc2darray(gridX - 1, gridY);
		DbBy_join = Malloc2darray(gridX - 1, gridY);

		//allocate metamaterial constants=============
		am1_x_join_right = Malloc2darray(gridX, gridY - 1);
		am2_x_join_right = Malloc2darray(gridX, gridY - 1);
		bm0_x_join_right = Malloc2darray(gridX, gridY - 1);
		bm1_x_join_right = Malloc2darray(gridX, gridY - 1);
		bm2_x_join_right = Malloc2darray(gridX, gridY - 1);
		A_x_join_right = Malloc2darray(gridX, gridY - 1);

		am1_y_join_right = Malloc2darray(gridX - 1, gridY);
		am2_y_join_right = Malloc2darray(gridX - 1, gridY);
		bm0_y_join_right = Malloc2darray(gridX - 1, gridY);
		bm1_y_join_right = Malloc2darray(gridX - 1, gridY);
		bm2_y_join_right = Malloc2darray(gridX - 1, gridY);
		A_y_join_right = Malloc2darray(gridX - 1, gridY);

		cm1_x_join_right = Malloc2darray(gridX, gridY - 1);
		cm2_x_join_right = Malloc2darray(gridX, gridY - 1);
		dm0_x_join_right = Malloc2darray(gridX, gridY - 1);
		dm1_x_join_right = Malloc2darray(gridX, gridY - 1);
		dm2_x_join_right = Malloc2darray(gridX, gridY - 1);
		C_x_join_right = Malloc2darray(gridX, gridY - 1);

		cm1_y_join_right = Malloc2darray(gridX - 1, gridY);
		cm2_y_join_right = Malloc2darray(gridX - 1, gridY);
		dm0_y_join_right = Malloc2darray(gridX - 1, gridY);
		dm1_y_join_right = Malloc2darray(gridX - 1, gridY);
		dm2_y_join_right = Malloc2darray(gridX - 1, gridY);
		C_y_join_right = Malloc2darray(gridX - 1, gridY);

		//allocate Eps and Mu=========================
		Eps = Malloc3darray(gridX + 1, gridY + 1, sliceSize + 1);
		Mu = Malloc3darray(gridX + 1, gridY + 1, sliceSize + 1);

		//allocate Sigma==============================
		Sigma = Malloc3darray(gridX + 1, gridY + 1, sliceSize + 1);
		Sigma1 = Malloc3darray(gridX + 1, gridY + 1, sliceSize + 1);

		//allocate omega and gamma====================
		omegaEP = Malloc3darray(gridX + 1, gridY + 1, sliceSize + 1);
		omegaMP = Malloc3darray(gridX + 1, gridY + 1, sliceSize + 1);
		gammaE = Malloc3darray(gridX + 1, gridY + 1, sliceSize + 1);
		gammaM = Malloc3darray(gridX + 1, gridY + 1, sliceSize + 1);	
	}
	else
	{
		//allocate Eps and Mu=========================
		Eps = Malloc3darray(gridX + 1, gridY + 1, sliceSize + 1);
		Mu = Malloc3darray(gridX + 1, gridY + 1, sliceSize + 1);

		//allocate Sigma==============================
		Sigma = Malloc3darray(gridX + 1, gridY + 1, sliceSize + 1);
		Sigma1 = Malloc3darray(gridX + 1, gridY + 1, sliceSize + 1);

		//allocate omega and gamma====================
		omegaEP = Malloc3darray(gridX + 1, gridY + 1, sliceSize + 1);
		omegaMP = Malloc3darray(gridX + 1, gridY + 1, sliceSize + 1);
		gammaE = Malloc3darray(gridX + 1, gridY + 1, sliceSize + 1);
		gammaM = Malloc3darray(gridX + 1, gridY + 1, sliceSize + 1);
	}

	//allocate Ca, Cb, Cc for E=======================
	CaEx = Malloc3darray(gridX - 1, gridY, sliceSize);
	CbEx = Malloc3darray(gridX - 1, gridY, sliceSize);
	CcEx = Malloc3darray(gridX - 1, gridY, sliceSize);
	CaEy = Malloc3darray(gridX, gridY - 1, sliceSize);
	CbEy = Malloc3darray(gridX, gridY - 1, sliceSize);
	CcEy = Malloc3darray(gridX, gridY - 1, sliceSize);
	CaEz = Malloc3darray(gridX, gridY, sliceSize - 1);
	CbEz = Malloc3darray(gridX, gridY, sliceSize - 1);
	CcEz = Malloc3darray(gridX, gridY, sliceSize - 1);

	//allocate Da, Db, Dc for H=======================
	DaHx = Malloc3darray(gridX - 1, gridY, sliceSize);
	DbHx = Malloc3darray(gridX - 1, gridY, sliceSize);
	DcHx = Malloc3darray(gridX - 1, gridY, sliceSize);
	DaHy = Malloc3darray(gridX, gridY - 1, sliceSize);
	DbHy = Malloc3darray(gridX, gridY - 1, sliceSize);
	DcHy = Malloc3darray(gridX, gridY - 1, sliceSize);
	DaHz = Malloc3darray(gridX, gridY, sliceSize - 1);
	DbHz = Malloc3darray(gridX, gridY, sliceSize - 1);
	DcHz = Malloc3darray(gridX, gridY, sliceSize - 1);

	//allocate Ca, Cb for D===========================
	CaDx = Malloc3darray(gridX, gridY - 1, sliceSize - 1);
	CbDx = Malloc3darray(gridX, gridY - 1, sliceSize - 1);
	CaDy = Malloc3darray(gridX - 1, gridY, sliceSize - 1);
	CbDy = Malloc3darray(gridX - 1, gridY, sliceSize - 1);
	CaDz = Malloc3darray(gridX - 1, gridY - 1, sliceSize);
	CbDz = Malloc3darray(gridX - 1, gridY - 1, sliceSize);

	//allocate Ca, Cb, Da, Db for B===================
	DaBx = Malloc3darray(gridX, gridY - 1, sliceSize - 1);
	DbBx = Malloc3darray(gridX, gridY - 1, sliceSize - 1);
	DaBy = Malloc3darray(gridX - 1, gridY, sliceSize - 1);
	DbBy = Malloc3darray(gridX - 1, gridY, sliceSize - 1);
	DaBz = Malloc3darray(gridX - 1, gridY - 1, sliceSize);
	DbBz = Malloc3darray(gridX - 1, gridY - 1, sliceSize);

	//allocate metamaterial constants for E=========== 
	am1_x = Malloc3darray(gridX, gridY - 1, sliceSize - 1);
	am2_x = Malloc3darray(gridX, gridY - 1, sliceSize - 1);
	A_x = Malloc3darray(gridX, gridY - 1, sliceSize - 1);
	am1_y = Malloc3darray(gridX - 1, gridY, sliceSize - 1);
	am2_y = Malloc3darray(gridX - 1, gridY, sliceSize - 1);
	A_y = Malloc3darray(gridX - 1, gridY, sliceSize - 1);
	am1_z = Malloc3darray(gridX - 1, gridY - 1, sliceSize);
	am2_z = Malloc3darray(gridX - 1, gridY - 1, sliceSize);
	A_z = Malloc3darray(gridX - 1, gridY - 1, sliceSize);

	bm0_x = Malloc3darray(gridX, gridY - 1, sliceSize - 1);
	bm1_x = Malloc3darray(gridX, gridY - 1, sliceSize - 1);
	bm2_x = Malloc3darray(gridX, gridY - 1, sliceSize - 1);
	bm0_y = Malloc3darray(gridX - 1, gridY, sliceSize - 1);
	bm1_y = Malloc3darray(gridX - 1, gridY, sliceSize - 1);
	bm2_y = Malloc3darray(gridX - 1, gridY, sliceSize - 1);
	bm0_z = Malloc3darray(gridX - 1, gridY - 1, sliceSize);
	bm1_z = Malloc3darray(gridX - 1, gridY - 1, sliceSize);
	bm2_z = Malloc3darray(gridX - 1, gridY - 1, sliceSize);

	//allocate metamaterial constants for H=========== 
	cm1_x = Malloc3darray(gridX, gridY - 1, sliceSize - 1);
	cm2_x = Malloc3darray(gridX, gridY - 1, sliceSize - 1);
	C_x = Malloc3darray(gridX, gridY - 1, sliceSize - 1);
	cm1_y = Malloc3darray(gridX - 1, gridY, sliceSize - 1);
	cm2_y = Malloc3darray(gridX - 1, gridY, sliceSize - 1);
	C_y = Malloc3darray(gridX - 1, gridY, sliceSize - 1);
	cm1_z = Malloc3darray(gridX - 1, gridY - 1, sliceSize);
	cm2_z = Malloc3darray(gridX - 1, gridY - 1, sliceSize);
	C_z = Malloc3darray(gridX - 1, gridY - 1, sliceSize);

	dm0_x = Malloc3darray(gridX, gridY - 1, sliceSize - 1);
	dm1_x = Malloc3darray(gridX, gridY - 1, sliceSize - 1);
	dm2_x = Malloc3darray(gridX, gridY - 1, sliceSize - 1);
	dm0_y = Malloc3darray(gridX - 1, gridY, sliceSize - 1);
	dm1_y = Malloc3darray(gridX - 1, gridY, sliceSize - 1);
	dm2_y = Malloc3darray(gridX - 1, gridY, sliceSize - 1);
	dm0_z = Malloc3darray(gridX - 1, gridY - 1, sliceSize);
	dm1_z = Malloc3darray(gridX - 1, gridY - 1, sliceSize);
	dm2_z = Malloc3darray(gridX - 1, gridY - 1, sliceSize);

	//allocate buffers
	Ex_buf = Malloc2darray(gridX - 1, gridY);
	Ey_buf = Malloc2darray(gridX, gridY - 1);

	Hx_buf = Malloc2darray(gridX - 1, gridY);
	Hy_buf = Malloc2darray(gridX, gridY - 1);

	Dx_buf = Malloc2darray(gridX, gridY - 1);
	Dy_buf = Malloc2darray(gridX - 1, gridY);
	Dz_buf = Malloc2darray(gridX - 1, gridY - 1);
	Dx_buf_prev = Malloc2darray(gridX, gridY - 1);
	Dy_buf_prev = Malloc2darray(gridX - 1, gridY);
	Dz_buf_prev = Malloc2darray(gridX - 1, gridY - 1);

	Bx_buf = Malloc2darray(gridX, gridY - 1);
	By_buf = Malloc2darray(gridX - 1, gridY);
	Bz_buf = Malloc2darray(gridX - 1, gridY - 1);
	Bx_buf_prev = Malloc2darray(gridX, gridY - 1);
	By_buf_prev = Malloc2darray(gridX - 1, gridY);
	Bz_buf_prev = Malloc2darray(gridX - 1, gridY - 1);

	bufx = new double[gridX*gridY];
	bufy = new double[gridX*gridY];
	bufz = new double[gridX*gridY];
	bufk = new double[gridX*gridY];

	rbufx = new double[gridX*gridY];
	rbufy = new double[gridX*gridY];
	rbufz = new double[gridX*gridY];
	rbufk = new double[gridX*gridY];

	if (calculateAmplitude)
	{
		Ex_amp = Malloc3darray(gridX - 1, gridY, sliceSize);
		Ey_amp = Malloc3darray(gridX, gridY - 1, sliceSize);
		Ez_amp = Malloc3darray(gridX, gridY, sliceSize - 1);

		Hx_amp = Malloc3darray(gridX - 1, gridY, sliceSize);
		Hy_amp = Malloc3darray(gridX, gridY - 1, sliceSize);
		Hz_amp = Malloc3darray(gridX, gridY, sliceSize - 1);

		if (rank != numProcs - 1)
		{
			Ex_join_right_amp = Malloc2darray(gridX - 1, gridY);
			Ey_join_right_amp = Malloc2darray(gridX, gridY - 1);
			Hz_join_right_amp = Malloc2darray(gridX, gridY);
		}
	}

	if (rank == 0)
	{
		timeAmp = new double[numTimeSteps + amplitudeSteps];
	}
}
void freeFields()
{
	Dealloc3darray(Ex, gridX - 1, gridY, sliceSize);
	Dealloc3darray(Ey, gridX, gridY - 1, sliceSize);
	Dealloc3darray(Ez, gridX, gridY, sliceSize - 1);

	Dealloc3darray(Hx, gridX - 1, gridY, sliceSize);
	Dealloc3darray(Hy, gridX, gridY - 1, sliceSize);
	Dealloc3darray(Hz, gridX, gridY, sliceSize - 1);
}
void freeAllExceptFields()
{
	//delete[] E_inc;
	//delete[] H_inc;

	delete[] W1;
	delete[] Wstep;

	//deallocate D==========================================
	Dealloc3darray(Dx, gridX, gridY - 1, sliceSize - 1);
	Dealloc3darray(Dy, gridX - 1, gridY, sliceSize - 1);
	Dealloc3darray(Dz, gridX - 1, gridY - 1, sliceSize);

	Dealloc3darray(Dx_prev, gridX, gridY - 1, sliceSize - 1);
	Dealloc3darray(Dy_prev, gridX - 1, gridY, sliceSize - 1);
	Dealloc3darray(Dz_prev, gridX - 1, gridY - 1, sliceSize);

	Dealloc3darray(Dx_prev2, gridX, gridY - 1, sliceSize - 1);
	Dealloc3darray(Dy_prev2, gridX - 1, gridY, sliceSize - 1);
	Dealloc3darray(Dz_prev2, gridX - 1, gridY - 1, sliceSize);

	//deallocate D1=========================================
	Dealloc3darray(Dx1, gridX, gridY - 1, sliceSize - 1);
	Dealloc3darray(Dy1, gridX - 1, gridY, sliceSize - 1);
	Dealloc3darray(Dz1, gridX - 1, gridY - 1, sliceSize);

	Dealloc3darray(Dx1_prev, gridX, gridY - 1, sliceSize - 1);
	Dealloc3darray(Dy1_prev, gridX - 1, gridY, sliceSize - 1);
	Dealloc3darray(Dz1_prev, gridX - 1, gridY - 1, sliceSize);

	Dealloc3darray(Dx1_prev2, gridX, gridY - 1, sliceSize - 1);
	Dealloc3darray(Dy1_prev2, gridX - 1, gridY, sliceSize - 1);
	Dealloc3darray(Dz1_prev2, gridX - 1, gridY - 1, sliceSize);

	//deallocate B=========================================
	Dealloc3darray(Bx, gridX, gridY - 1, sliceSize - 1);
	Dealloc3darray(By, gridX - 1, gridY, sliceSize - 1);
	Dealloc3darray(Bz, gridX - 1, gridY - 1, sliceSize);

	Dealloc3darray(Bx_prev, gridX, gridY - 1, sliceSize - 1);
	Dealloc3darray(By_prev, gridX - 1, gridY, sliceSize - 1);
	Dealloc3darray(Bz_prev, gridX - 1, gridY - 1, sliceSize);

	Dealloc3darray(Bx_prev2, gridX, gridY - 1, sliceSize - 1);
	Dealloc3darray(By_prev2, gridX - 1, gridY, sliceSize - 1);
	Dealloc3darray(Bz_prev2, gridX - 1, gridY - 1, sliceSize);

	//deallocate B1========================================
	Dealloc3darray(Bx1, gridX, gridY - 1, sliceSize - 1);
	Dealloc3darray(By1, gridX - 1, gridY, sliceSize - 1);
	Dealloc3darray(Bz1, gridX - 1, gridY - 1, sliceSize);

	Dealloc3darray(Bx1_prev, gridX, gridY - 1, sliceSize - 1);
	Dealloc3darray(By1_prev, gridX - 1, gridY, sliceSize - 1);
	Dealloc3darray(Bz1_prev, gridX - 1, gridY - 1, sliceSize);

	Dealloc3darray(Bx1_prev2, gridX, gridY - 1, sliceSize - 1);
	Dealloc3darray(By1_prev2, gridX - 1, gridY, sliceSize - 1);
	Dealloc3darray(Bz1_prev2, gridX - 1, gridY - 1, sliceSize);

	//join deallocation====================================
	if (rank != numProcs - 1)
	{
		//deallocate fields===============================
		Dealloc2darray(Ez_join_right, gridX, gridY);
		Dealloc2darray(Hz_join_right, gridX, gridY);

		//deallocate D and B==============================
		Dealloc2darray(Dx_join_right, gridX, gridY - 1);
		Dealloc2darray(Dy_join_right, gridX - 1, gridY);
		Dealloc2darray(Bx_join_right, gridX, gridY - 1);
		Dealloc2darray(By_join_right, gridX - 1, gridY);

		Dealloc2darray(Dx_join_right_prev, gridX, gridY - 1);
		Dealloc2darray(Dy_join_right_prev, gridX - 1, gridY);
		Dealloc2darray(Bx_join_right_prev, gridX, gridY - 1);
		Dealloc2darray(By_join_right_prev, gridX - 1, gridY);

		Dealloc2darray(Dx_join_right_prev2, gridX, gridY - 1);
		Dealloc2darray(Dy_join_right_prev2, gridX - 1, gridY);
		Dealloc2darray(Bx_join_right_prev2, gridX, gridY - 1);
		Dealloc2darray(By_join_right_prev2, gridX - 1, gridY);

		//deallocate D1 and B1============================
		Dealloc2darray(Dx1_join_right, gridX, gridY - 1);
		Dealloc2darray(Dy1_join_right, gridX - 1, gridY);
		Dealloc2darray(Bx1_join_right, gridX, gridY - 1);
		Dealloc2darray(By1_join_right, gridX - 1, gridY);

		Dealloc2darray(Dx1_join_right_prev, gridX, gridY - 1);
		Dealloc2darray(Dy1_join_right_prev, gridX - 1, gridY);
		Dealloc2darray(Bx1_join_right_prev, gridX, gridY - 1);
		Dealloc2darray(By1_join_right_prev, gridX - 1, gridY);

		Dealloc2darray(Dx1_join_right_prev2, gridX, gridY - 1);
		Dealloc2darray(Dy1_join_right_prev2, gridX - 1, gridY);
		Dealloc2darray(Bx1_join_right_prev2, gridX, gridY - 1);
		Dealloc2darray(By1_join_right_prev2, gridX - 1, gridY);

		//deallocate Ca, Cb, Cc, Da, Db, Dc for fields===
		Dealloc2darray(CaEz_join, gridX, gridY);
		Dealloc2darray(CbEz_join, gridX, gridY);
		Dealloc2darray(CcEz_join, gridX, gridY);
		Dealloc2darray(DaHz_join, gridX, gridY);
		Dealloc2darray(DbHz_join, gridX, gridY);
		Dealloc2darray(DcHz_join, gridX, gridY);

		//deallocate Ca, Cb, Da, Db for D and B==========
		Dealloc2darray(CaDx_join, gridX, gridY - 1);
		Dealloc2darray(CbDx_join, gridX, gridY - 1);
		Dealloc2darray(CaDy_join, gridX - 1, gridY);
		Dealloc2darray(CbDy_join, gridX - 1, gridY);
		Dealloc2darray(DaBx_join, gridX, gridY - 1);
		Dealloc2darray(DbBx_join, gridX, gridY - 1);
		Dealloc2darray(DaBy_join, gridX - 1, gridY);
		Dealloc2darray(DbBy_join, gridX - 1, gridY);

		//deallocate metamaterial constants=============
		Dealloc2darray(am1_x_join_right, gridX, gridY - 1);
		Dealloc2darray(am2_x_join_right, gridX, gridY - 1);
		Dealloc2darray(A_x_join_right, gridX, gridY - 1);
		Dealloc2darray(bm0_x_join_right, gridX, gridY - 1);
		Dealloc2darray(bm1_x_join_right, gridX, gridY - 1);
		Dealloc2darray(bm2_x_join_right, gridX, gridY - 1);

		Dealloc2darray(am1_y_join_right, gridX - 1, gridY);
		Dealloc2darray(am2_y_join_right, gridX - 1, gridY);
		Dealloc2darray(A_y_join_right, gridX - 1, gridY);
		Dealloc2darray(bm0_y_join_right, gridX - 1, gridY);
		Dealloc2darray(bm1_y_join_right, gridX - 1, gridY);
		Dealloc2darray(bm2_y_join_right, gridX - 1, gridY);

		Dealloc2darray(cm1_x_join_right, gridX, gridY - 1);
		Dealloc2darray(cm2_x_join_right, gridX, gridY - 1);
		Dealloc2darray(C_x_join_right, gridX, gridY - 1);
		Dealloc2darray(dm0_x_join_right, gridX, gridY - 1);
		Dealloc2darray(dm1_x_join_right, gridX, gridY - 1);
		Dealloc2darray(dm2_x_join_right, gridX, gridY - 1);

		Dealloc2darray(cm1_y_join_right, gridX - 1, gridY);
		Dealloc2darray(cm2_y_join_right, gridX - 1, gridY);
		Dealloc2darray(C_y_join_right, gridX - 1, gridY);
		Dealloc2darray(dm0_y_join_right, gridX - 1, gridY);
		Dealloc2darray(dm1_y_join_right, gridX - 1, gridY);
		Dealloc2darray(dm2_y_join_right, gridX - 1, gridY);

		//deallocate Eps and Mu=========================
		Dealloc3darray(Eps, gridX + 1, gridY + 1, sliceSize + 1);
		Dealloc3darray(Mu, gridX + 1, gridY + 1, sliceSize + 1);

		//deallocate Sigma==============================
		Dealloc3darray(Sigma, gridX + 1, gridY + 1, sliceSize + 1);
		Dealloc3darray(Sigma1, gridX + 1, gridY + 1, sliceSize + 1);

		//deallocate omega and gamma====================
		Dealloc3darray(omegaEP, gridX + 1, gridY + 1, sliceSize + 1);
		Dealloc3darray(omegaMP, gridX + 1, gridY + 1, sliceSize + 1);
		Dealloc3darray(gammaE, gridX + 1, gridY + 1, sliceSize + 1);
		Dealloc3darray(gammaM, gridX + 1, gridY + 1, sliceSize + 1);
	}
	else
	{
		//deallocate Eps and Mu=========================
		Dealloc3darray(Eps, gridX + 1, gridY + 1, sliceSize + 1);
		Dealloc3darray(Mu, gridX + 1, gridY + 1, sliceSize + 1);

		//deallocate Sigma==============================
		Dealloc3darray(Sigma, gridX + 1, gridY + 1, sliceSize + 1);
		Dealloc3darray(Sigma1, gridX + 1, gridY + 1, sliceSize + 1);

		//deallocate omega and gamma====================
		Dealloc3darray(omegaEP, gridX + 1, gridY + 1, sliceSize + 1);
		Dealloc3darray(omegaMP, gridX + 1, gridY + 1, sliceSize + 1);
		Dealloc3darray(gammaE, gridX + 1, gridY + 1, sliceSize + 1);
		Dealloc3darray(gammaM, gridX + 1, gridY + 1, sliceSize + 1);
	}

	//deallocate Ca, Cb, Cc for E=======================
	Dealloc3darray(CaEx, gridX - 1, gridY, sliceSize);
	Dealloc3darray(CbEx, gridX - 1, gridY, sliceSize);
	Dealloc3darray(CcEx, gridX - 1, gridY, sliceSize);
	Dealloc3darray(CaEy, gridX, gridY - 1, sliceSize);
	Dealloc3darray(CbEy, gridX, gridY - 1, sliceSize);
	Dealloc3darray(CcEy, gridX, gridY - 1, sliceSize);
	Dealloc3darray(CaEz, gridX, gridY, sliceSize - 1);
	Dealloc3darray(CbEz, gridX, gridY, sliceSize - 1);
	Dealloc3darray(CcEz, gridX, gridY, sliceSize - 1);

	//deallocate Da, Db, Dc for H=======================
	Dealloc3darray(DaHx, gridX - 1, gridY, sliceSize);
	Dealloc3darray(DbHx, gridX - 1, gridY, sliceSize);
	Dealloc3darray(DcHx, gridX - 1, gridY, sliceSize);
	Dealloc3darray(DaHy, gridX, gridY - 1, sliceSize);
	Dealloc3darray(DbHy, gridX, gridY - 1, sliceSize);
	Dealloc3darray(DcHy, gridX, gridY - 1, sliceSize);
	Dealloc3darray(DaHz, gridX, gridY, sliceSize - 1);
	Dealloc3darray(DbHz, gridX, gridY, sliceSize - 1);
	Dealloc3darray(DcHz, gridX, gridY, sliceSize - 1);

	//dellocate Ca, Cb for D===========================
	Dealloc3darray(CaDx, gridX, gridY - 1, sliceSize - 1);
	Dealloc3darray(CbDx, gridX, gridY - 1, sliceSize - 1);
	Dealloc3darray(CaDy, gridX - 1, gridY, sliceSize - 1);
	Dealloc3darray(CbDy, gridX - 1, gridY, sliceSize - 1);
	Dealloc3darray(CaDz, gridX - 1, gridY - 1, sliceSize);
	Dealloc3darray(CbDz, gridX - 1, gridY - 1, sliceSize);

	//deallocate Ca, Cb, Da, Db for B===================
	Dealloc3darray(DaBx, gridX, gridY - 1, sliceSize - 1);
	Dealloc3darray(DbBx, gridX, gridY - 1, sliceSize - 1);
 	Dealloc3darray(DaBy, gridX - 1, gridY, sliceSize - 1);
	Dealloc3darray(DbBy, gridX - 1, gridY, sliceSize - 1);
	Dealloc3darray(DaBz, gridX - 1, gridY - 1, sliceSize);
	Dealloc3darray(DbBz, gridX - 1, gridY - 1, sliceSize);

	//deallocate metamaterial constants for E=========== 
	Dealloc3darray(am1_x, gridX, gridY - 1, sliceSize - 1);
	Dealloc3darray(am2_x, gridX, gridY - 1, sliceSize - 1);
	Dealloc3darray(A_x, gridX, gridY - 1, sliceSize - 1);
	Dealloc3darray(am1_y, gridX - 1, gridY, sliceSize - 1);
	Dealloc3darray(am2_y, gridX - 1, gridY, sliceSize - 1);
	Dealloc3darray(A_y, gridX - 1, gridY, sliceSize - 1);
	Dealloc3darray(am1_z, gridX - 1, gridY - 1, sliceSize);
	Dealloc3darray(am2_z, gridX - 1, gridY - 1, sliceSize);
	Dealloc3darray(A_z, gridX - 1, gridY - 1, sliceSize);

	Dealloc3darray(bm0_x, gridX, gridY - 1, sliceSize - 1);
	Dealloc3darray(bm1_x, gridX, gridY - 1, sliceSize - 1);
	Dealloc3darray(bm2_x, gridX, gridY - 1, sliceSize - 1);
	Dealloc3darray(bm0_y, gridX - 1, gridY, sliceSize - 1);
	Dealloc3darray(bm1_y, gridX - 1, gridY, sliceSize - 1);
	Dealloc3darray(bm2_y, gridX - 1, gridY, sliceSize - 1);
	Dealloc3darray(bm0_z, gridX - 1, gridY - 1, sliceSize);
	Dealloc3darray(bm1_z, gridX - 1, gridY - 1, sliceSize);
	Dealloc3darray(bm2_z, gridX - 1, gridY - 1, sliceSize);

	//deallocate metamaterial constants for H=========== 
	Dealloc3darray(cm1_x, gridX, gridY - 1, sliceSize - 1);
	Dealloc3darray(cm2_x, gridX, gridY - 1, sliceSize - 1);
	Dealloc3darray(C_x, gridX, gridY - 1, sliceSize - 1);
	Dealloc3darray(cm1_y, gridX - 1, gridY, sliceSize - 1);
	Dealloc3darray(cm2_y, gridX - 1, gridY, sliceSize - 1);
	Dealloc3darray(C_y, gridX - 1, gridY, sliceSize - 1);
	Dealloc3darray(cm1_z, gridX - 1, gridY - 1, sliceSize);
	Dealloc3darray(cm2_z, gridX - 1, gridY - 1, sliceSize);
	Dealloc3darray(C_z, gridX - 1, gridY - 1, sliceSize);

	Dealloc3darray(dm0_x, gridX, gridY - 1, sliceSize - 1);
	Dealloc3darray(dm1_x, gridX, gridY - 1, sliceSize - 1);
	Dealloc3darray(dm2_x, gridX, gridY - 1, sliceSize - 1);
	Dealloc3darray(dm0_y, gridX - 1, gridY, sliceSize - 1);
	Dealloc3darray(dm1_y, gridX - 1, gridY, sliceSize - 1);
	Dealloc3darray(dm2_y, gridX - 1, gridY, sliceSize - 1);
	Dealloc3darray(dm0_z, gridX - 1, gridY - 1, sliceSize);
	Dealloc3darray(dm1_z, gridX - 1, gridY - 1, sliceSize);
	Dealloc3darray(dm2_z, gridX - 1, gridY - 1, sliceSize);

	//allocate buffers
	Dealloc2darray(Ex_buf, gridX - 1, gridY);
	Dealloc2darray(Ey_buf, gridX, gridY - 1);

	Dealloc2darray(Hx_buf, gridX - 1, gridY);
	Dealloc2darray(Hy_buf, gridX, gridY - 1);

	Dealloc2darray(Dx_buf, gridX, gridY - 1);
	Dealloc2darray(Dy_buf, gridX - 1, gridY);
	Dealloc2darray(Dz_buf, gridX - 1, gridY - 1);
	Dealloc2darray(Dx_buf_prev, gridX, gridY - 1);
	Dealloc2darray(Dy_buf_prev, gridX - 1, gridY);
	Dealloc2darray(Dz_buf_prev, gridX - 1, gridY - 1);

	Dealloc2darray(Bx_buf, gridX, gridY - 1);
	Dealloc2darray(By_buf, gridX - 1, gridY);

	delete[] bufx;
	delete[] bufy;
	delete[] bufz;
	delete[] bufk;

	delete[] rbufx;
	delete[] rbufy;
	delete[] rbufz;
	delete[] rbufk;

	if (calculateAmplitude)
	{
		Dealloc3darray(Ex_amp, gridX - 1, gridY, sliceSize);
		Dealloc3darray(Ey_amp, gridX, gridY - 1, sliceSize);
		Dealloc3darray(Ez_amp, gridX, gridY, sliceSize - 1);

		Dealloc3darray(Hx_amp, gridX - 1, gridY, sliceSize);
		Dealloc3darray(Hy_amp, gridX, gridY - 1, sliceSize);
		Dealloc3darray(Hz_amp, gridX, gridY, sliceSize - 1);

		if (rank != numProcs - 1)
		{
			Dealloc2darray(Ex_join_right_amp, gridX - 1, gridY);
			Dealloc2darray(Ey_join_right_amp, gridX, gridY - 1);
			Dealloc2darray(Hz_join_right_amp, gridX, gridY);
		}
	}

	if (rank == 0)
	{
		delete[] timeAmp;
	}
}
void freeAll()
{
	freeFields();
	freeAllExceptFields();
}

//Initialize everything
void getGammaOmega(int num, double &omega, double &gamma, double om1, double om2, double g1, double g2,
	double om3 = 0.0, double om4 = 0.0, double g3 = 0.0, double g4 = 0.0)
{
	double A1 = om1*om1*(OMEGA*OMEGA + g2*g2)*(OMEGA*OMEGA + g3*g3)*(OMEGA*OMEGA + g4*g4) +
		om2*om2*(OMEGA*OMEGA + g1*g1)*(OMEGA*OMEGA + g3*g3)*(OMEGA*OMEGA + g4*g4) +
		om3*om3*(OMEGA*OMEGA + g1*g1)*(OMEGA*OMEGA + g2*g2)*(OMEGA*OMEGA + g4*g4) +
		om4*om4*(OMEGA*OMEGA + g1*g1)*(OMEGA*OMEGA + g2*g2)*(OMEGA*OMEGA + g3*g3);
	double B1 = (OMEGA*OMEGA + g1*g1)*(OMEGA*OMEGA + g2*g2)*(OMEGA*OMEGA + g3*g3)*(OMEGA*OMEGA + g4*g4);
	
	double A2 = om1*om1*g1*(OMEGA*OMEGA + g2*g2)*(OMEGA*OMEGA + g3*g3)*(OMEGA*OMEGA + g4*g4) +
		om2*om2*g2*(OMEGA*OMEGA + g1*g1)*(OMEGA*OMEGA + g3*g3)*(OMEGA*OMEGA + g4*g4) +
		om3*om3*g3*(OMEGA*OMEGA + g1*g1)*(OMEGA*OMEGA + g2*g2)*(OMEGA*OMEGA + g4*g4) +
		om4*om4*g4*(OMEGA*OMEGA + g1*g1)*(OMEGA*OMEGA + g2*g2)*(OMEGA*OMEGA + g3*g3);
	double B2 = B1 * OMEGA;

	double tmpC1 = A1 / B1;
	double tmpC2 = A2 / B2;

	if (tmpC1 == 0)
	{
		gamma = (g1 + g2 + g3 + g4) / num;
		omega = 0.0;
	}
	else
	{
		gamma = OMEGA * tmpC2 / tmpC1;
		omega = OMEGA * sqrtf((tmpC1*tmpC1 + tmpC2*tmpC2) / (num * tmpC1));
	}
	
	//gamma = (g1 + g2 + g3 + g4) / num;
	//omega = (om1 + om2 + om3 + om4) / num;

	if (om1 == 0 && g1 > 0 || om2 == 0 && g2 > 0 || om3 == 0 && g3 > 0 || om4 == 0 && g4 > 0)
	{
		gamma = (g1 + g2 + g3 + g4) / num;
	}
}

void initializeAll()
{
	double boundary = pmlL*delta;

	//ka_max = 1.0;
	exponent = 6;
	double R_err = 1e-16;
	double eta_1 = sqrt(MU_Z / EPS_Z);
	double eta_2 = sqrt(MU_Z / EPS_Z);
	double sigma_max_1 = -log(R_err) * (exponent + 1.0) / (2.0 * IMP_Z * boundary);
	double sigma_max_2 = -log(R_err) * (exponent + 1.0) / (2.0 * IMP_Z * boundary);
	boundaryFactor = sigma_max_1 / (dx * (pow(boundary, exponent)) * (exponent + 1));

	int endK = sliceSize;
	//if (rank == numProcs - 1)
	//	--endK;

	double addition = rank * sliceSize;

	for (int i = 0; i <= gridX; ++i)
	for (int j = 0; j <= gridY; ++j)
	{
		for (int k = 0; k <= endK; ++k)
		{
			Eps[i][j][k] = 1;
			Mu[i][j][k] = 1;
			Sigma[i][j][k] = 0.0;
			Sigma1[i][j][k] = 0.0;

			omegaEP[i][j][k] = 0.0;
			omegaMP[i][j][k] = 0.0;
			gammaE[i][j][k] = 0.0;
			gammaM[i][j][k] = 0.0;

			/*int startK = 0;
			int endK = 0;
			if (k >= startK && k < sliceSize - endK &&
				j >= pmlL && j < gridY - pmlR &&
				i >= 3*gridX / 8 + 1 && i <= 3*gridX / 8 + gridX/4 + 1)
			{
				double tmpG = 0.02 * OMEGA / (-2);
				double tmpTau = -1 / tmpG;
				omegaEP[i][j][k] = sqrtf(2.0) *OMEGA;//sqrtf(0.02*OMEGA*(OMEGA*OMEGA*tmpTau*tmpTau+1)/tmpTau);
				gammaE[i][j][k] = 0.01 * OMEGA;// tmpG;
				omegaMP[i][j][k] = sqrtf(2.0) *OMEGA;// sqrtf(0.02*OMEGA*(OMEGA*OMEGA*tmpTau*tmpTau + 1) / tmpTau);
				gammaM[i][j][k] = 0.01 * OMEGA;//tmpG;
				if ((i - gridX / 2 - 1) * (i - gridX / 2 - 1) + (j - gridY / 2 - 1) * (j - gridY / 2 - 1) < 41)
				{
					omegaEP[i][j][k] = sqrtf(2.0) *OMEGA;
					gammaE[i][j][k] = 0.5 * OMEGA;
					omegaMP[i][j][k] = 0.0;// sqrtf(2.0) *OMEGA;
					gammaM[i][j][k] = 0.0; // 0.5 * OMEGA;
				}
				//Eps[i][j][k] = 2.0;
			}*/
		}
	}
	std::cout << "Init material: rank " << rank << std::endl;

	//Ex
	for (int i = 0; i <= gridX - 2; ++i)
	for (int j = 0; j <= gridY - 1; ++j)
	for (int k = 0; k <= sliceSize - 1; ++k)
	{
		double posX = i + 1.5;
		double posY = j + 1;
		double posZ = addition + k + 1;

		double tmpEps = (Eps[i + 1][j + 1][k + 1] + Eps[i + 1][j][k + 1] +
			Eps[i + 1][j + 1][k] + Eps[i + 1][j][k]) / 4;

		CaEx[i][j][k] = (2 * EPS_Z * Kz(posZ) - SigmaZ(posZ) * dt) /
			(2 * EPS_Z * Kz(posZ) + SigmaZ(posZ) * dt);
		CbEx[i][j][k] = (2 * EPS_Z * Kx(posX) + SigmaX(posX, posY) * dt) /
			((2 * EPS_Z * Kz(posZ) + SigmaZ(posZ) * dt) * tmpEps * EPS_Z);
		CcEx[i][j][k] = (2 * EPS_Z * Kx(posX) - SigmaX(posX, posY) * dt) /
			((2 * EPS_Z * Kz(posZ) + SigmaZ(posZ) * dt) * tmpEps * EPS_Z);
	}

	//Dx
	for (int i = 0; i <= gridX - 1; ++i)
	for (int j = 0; j <= gridY - 2; ++j)
	for (int k = 0; k <= sliceSize - 2; ++k)
	{
		double posX = i + 1;
		double posY = j + 1.5;
		double posZ = addition + k + 1.5;

		CaDx[i][j][k] = (2 * EPS_Z * Ky(posY) - SigmaY(posX, posY) * dt) /
						(2 * EPS_Z * Ky(posY) + SigmaY(posX, posY) * dt);
		CbDx[i][j][k] = (2 * EPS_Z * dt) / 
						((2 * EPS_Z * Ky(posY) + SigmaY(posX, posY) * dt));

		double tmpGammaE;
		double tmpOmegaEP;
		getGammaOmega(2, tmpOmegaEP, tmpGammaE, omegaEP[i + 1][j + 1][k + 1], omegaEP[i][j + 1][k + 1],
			gammaE[i + 1][j + 1][k + 1], gammaE[i][j + 1][k + 1]);
	
		A_x[i][j][k] = 1 + dt*tmpGammaE / 2 + dt*dt*tmpOmegaEP*tmpOmegaEP / 4;
		am1_x[i][j][k] = (dt*dt*tmpOmegaEP*tmpOmegaEP / 2 - 2);
		am2_x[i][j][k] = (1 - dt*tmpGammaE / 2 + dt*dt*tmpOmegaEP*tmpOmegaEP / 4);
		bm0_x[i][j][k] = (1 + dt*tmpGammaE/2);
		bm1_x[i][j][k] = (-2);
		bm2_x[i][j][k] = (1 - dt*tmpGammaE/2);
	}
	
	std::cout << "Init Ex and Dx: rank " << rank << std::endl;

	//Ey
	for (int i = 0; i <= gridX - 1; ++i)
	for (int j = 0; j <= gridY - 2; ++j)
	for (int k = 0; k <= sliceSize - 1; ++k)
	{
		double posX = i + 1;
		double posY = j + 1.5;
		double posZ = addition + k + 1;

		double tmpEps = (Eps[i + 1][j + 1][k + 1] + Eps[i][j + 1][k + 1] +
			Eps[i + 1][j + 1][k] + Eps[i][j + 1][k]) / 4;

		CaEy[i][j][k] = (2 * EPS_Z * Kx(posX) - SigmaX(posX, posY) * dt) /
			(2 * EPS_Z * Kx(posX) + SigmaX(posX, posY) * dt);
		CbEy[i][j][k] = (2 * EPS_Z * Ky(posY) + SigmaY(posX, posY) * dt) /
			((2 * EPS_Z * Kx(posX) + SigmaX(posX, posY) * dt) * tmpEps * EPS_Z);
		CcEy[i][j][k] = (2 * EPS_Z * Ky(posY) - SigmaY(posX, posY) * dt) /
			((2 * EPS_Z * Kx(posX) + SigmaX(posX, posY) * dt) * tmpEps * EPS_Z);
	}

	//Dy
	for (int i = 0; i <= gridX - 2; ++i)
	for (int j = 0; j <= gridY - 1; ++j)
	for (int k = 0; k <= sliceSize - 2; ++k)
	{
		double posX = i + 1.5;
		double posY = j + 1;
		double posZ = addition + k + 1.5;

		CaDy[i][j][k] = (2 * EPS_Z * Kz(posZ) - SigmaZ(posZ) * dt) /
						(2 * EPS_Z * Kz(posZ) + SigmaZ(posZ) * dt);
		CbDy[i][j][k] = (2 * EPS_Z * dt) /
						((2 * EPS_Z * Kz(posZ) + SigmaZ(posZ) * dt));
		
		double tmpGammaE;
		double tmpOmegaEP;
		getGammaOmega(2, tmpOmegaEP, tmpGammaE, omegaEP[i + 1][j + 1][k + 1], omegaEP[i + 1][j][k + 1],
			gammaE[i + 1][j + 1][k + 1], gammaE[i + 1][j][k + 1]);

		A_y[i][j][k] = 1 + dt*tmpGammaE / 2 + dt*dt*tmpOmegaEP*tmpOmegaEP / 4;
		am1_y[i][j][k] = (dt*dt*tmpOmegaEP*tmpOmegaEP / 2 - 2);
		am2_y[i][j][k] = (1 - dt*tmpGammaE + dt*dt*tmpOmegaEP*tmpOmegaEP / 4);
		bm0_y[i][j][k] = (1 + dt*tmpGammaE/2);
		bm1_y[i][j][k] = (-2);
		bm2_y[i][j][k] = (1 - dt*tmpGammaE/2);
	}
	std::cout << "Init Ey and Dy: rank " << rank << std::endl;

	//Ez
	for (int i = 0; i <= gridX - 1; ++i)
	for (int j = 0; j <= gridY - 1; ++j)
	for (int k = 0; k <= sliceSize - 2; ++k)
	{
		double posX = i + 1;
		double posY = j + 1;
		double posZ = addition + k + 1.5;

		double tmpEps = (Eps[i + 1][j + 1][k + 1] + Eps[i][j + 1][k + 1] +
			Eps[i + 1][j][k + 1] + Eps[i][j][k + 1]) / 4;

		CaEz[i][j][k] = (2 * EPS_Z * Ky(posY) - SigmaY(posX, posY) * dt) /
			(2 * EPS_Z * Ky(posY) + SigmaY(posX, posY) * dt);
		CbEz[i][j][k] = (2 * EPS_Z * Kz(posZ) + SigmaZ(posZ) * dt) /
			((2 * EPS_Z * Ky(posY) + SigmaY(posX, posY) * dt) * tmpEps * EPS_Z);
		CcEz[i][j][k] = (2 * EPS_Z * Kz(posZ) - SigmaZ(posZ) * dt) /
			((2 * EPS_Z * Ky(posY) + SigmaY(posX, posY) * dt) * tmpEps * EPS_Z);
	}

	//Dz
	for (int i = 0; i <= gridX - 2; ++i)
	for (int j = 0; j <= gridY - 2; ++j)
	for (int k = 0; k <= sliceSize - 1; ++k)
	{
		double posX = i + 1.5;
		double posY = j + 1.5;
		double posZ = addition + k + 1;

		CaDz[i][j][k] = (2 * EPS_Z * Kx(posX) - SigmaX(posX, posY) * dt) /
						(2 * EPS_Z * Kx(posX) + SigmaX(posX, posY) * dt);
		CbDz[i][j][k] = (2 * EPS_Z * dt) /
						((2 * EPS_Z * Kx(posX) + SigmaX(posX, posY) * dt));

		double tmpGammaE;
		double tmpOmegaEP;
		getGammaOmega(2, tmpOmegaEP, tmpGammaE, omegaEP[i + 1][j + 1][k + 1], omegaEP[i + 1][j + 1][k],
			gammaE[i + 1][j + 1][k + 1], gammaE[i + 1][j + 1][k]);
		
		A_z[i][j][k] = 1 + dt*tmpGammaE + dt*dt*tmpOmegaEP*tmpOmegaEP / 2;
		am1_z[i][j][k] = (dt*dt*tmpOmegaEP*tmpOmegaEP / 2 - 2);
		am2_z[i][j][k] = (1 - dt*tmpGammaE + dt*dt*tmpOmegaEP*tmpOmegaEP / 4);
		bm0_z[i][j][k] = (1 + dt*tmpGammaE/2);
		bm1_z[i][j][k] = (-2);
		bm2_z[i][j][k] = (1 - dt*tmpGammaE/2);
	}
	std::cout << "Init Ez and Dz: rank " << rank << std::endl;

	//Hx
	for (int i = 0; i <= gridX - 2; ++i)
	for (int j = 0; j <= gridY - 1; ++j)
	for (int k = 0; k <= sliceSize - 1; ++k)
	{
		double posX = i + 1.5;
		double posY = j + 1;
		double posZ = addition + k + 1;

		double tmpMu = (Mu[i + 1][j + 1][k + 1] + Mu[i + 1][j][k + 1] +
			Mu[i + 1][j + 1][k] + Mu[i + 1][j][k]) / 4;

		DaHx[i][j][k] = (2 * EPS_Z * Kz(posZ) - SigmaZ(posZ) * dt) /
			(2 * EPS_Z * Kz(posZ) + SigmaZ(posZ) * dt);
		DbHx[i][j][k] = (2 * EPS_Z * Kx(posX) + SigmaX(posX, posY) * dt) /
			((2 * EPS_Z * Kz(posZ) + SigmaZ(posZ) * dt) * tmpMu * MU_Z);
		DcHx[i][j][k] = (2 * EPS_Z * Kx(posX) - SigmaX(posX, posY) * dt) /
			((2 * EPS_Z * Kz(posZ) + SigmaZ(posZ) * dt) * tmpMu * MU_Z);
	}

	//Bx
	for (int i = 0; i <= gridX - 1; ++i)
	for (int j = 0; j <= gridY - 2; ++j)
	for (int k = 0; k <= sliceSize - 2; ++k)
	{
		double posX = i + 1;
		double posY = j + 1.5;
		double posZ = addition + k + 1.5;

		DaBx[i][j][k] = (2 * EPS_Z * Ky(posY) - SigmaY(posX, posY) * dt) /
						(2 * EPS_Z * Ky(posY) + SigmaY(posX, posY) * dt);
		DbBx[i][j][k] = (2 * EPS_Z * dt) /
						((2 * EPS_Z * Ky(posY) + SigmaY(posX, posY) * dt));

		double tmpGammaM;
		double tmpOmegaMP;
		getGammaOmega(2, tmpOmegaMP, tmpGammaM, omegaMP[i + 1][j + 1][k + 1], omegaMP[i][j + 1][k + 1],
			gammaM[i + 1][j + 1][k + 1], gammaM[i][j + 1][k + 1]);

		C_x[i][j][k] = 1 + dt*tmpGammaM / 2 + dt*dt*tmpOmegaMP*tmpOmegaMP / 4;
		cm1_x[i][j][k] = (dt*dt*tmpOmegaMP*tmpOmegaMP / 2 - 2);
		cm2_x[i][j][k] = (1 - dt*tmpGammaM / 2 + dt*dt*tmpOmegaMP*tmpOmegaMP / 4);
		dm0_x[i][j][k] = (1 + dt*tmpGammaM/2);
		dm1_x[i][j][k] = (-2);
		dm2_x[i][j][k] = (1 - dt*tmpGammaM/2);
	}
	std::cout << "Init Hx and Bx: rank " << rank << std::endl;

	//Hy
	for (int i = 0; i <= gridX - 1; ++i)
	for (int j = 0; j <= gridY - 2; ++j)
	for (int k = 0; k <= sliceSize - 1; ++k)
	{
		double posX = i + 1;
		double posY = j + 1.5;
		double posZ = addition + k + 1;

		double tmpMu = (Mu[i + 1][j + 1][k + 1] + Mu[i][j + 1][k + 1] +
			Mu[i + 1][j + 1][k] + Mu[i][j + 1][k]) / 4;

		DaHy[i][j][k] = (2 * EPS_Z * Kx(posX) - SigmaX(posX, posY) * dt) /
			(2 * EPS_Z * Kx(posX) + SigmaX(posX, posY) * dt);
		DbHy[i][j][k] = (2 * EPS_Z * Ky(posY) + SigmaY(posX, posY) * dt) /
			((2 * EPS_Z * Kx(posX) + SigmaX(posX, posY) * dt) * tmpMu * MU_Z);
		DcHy[i][j][k] = (2 * EPS_Z * Ky(posY) - SigmaY(posX, posY) * dt) /
			((2 * EPS_Z * Kx(posX) + SigmaX(posX, posY) * dt) * tmpMu * MU_Z);
	}

	//By
	for (int i = 0; i <= gridX - 2; ++i)
	for (int j = 0; j <= gridY - 1; ++j)
	for (int k = 0; k <= sliceSize - 2; ++k)
	{
		double posX = i + 1.5;
		double posY = j + 1;
		double posZ = addition + k + 1.5;

		DaBy[i][j][k] = (2 * EPS_Z * Kz(posZ) - SigmaZ(posZ) * dt) /
						(2 * EPS_Z * Kz(posZ) + SigmaZ(posZ) * dt);
		DbBy[i][j][k] = (2 * EPS_Z * dt) /
						((2 * EPS_Z * Kz(posZ) + SigmaZ(posZ) * dt));

		double tmpGammaM;
		double tmpOmegaMP;
		getGammaOmega(4, tmpOmegaMP, tmpGammaM, omegaMP[i + 1][j + 1][k + 1], omegaMP[i + 1][j][k + 1],
			gammaM[i + 1][j + 1][k + 1], gammaM[i + 1][j][k + 1]);
		
		C_y[i][j][k] = 1 + dt*tmpGammaM / 2 + dt*dt*tmpOmegaMP*tmpOmegaMP / 4;
		cm1_y[i][j][k] = (dt*dt*tmpOmegaMP*tmpOmegaMP / 2 - 2);
		cm2_y[i][j][k] = (1 - dt*tmpGammaM / 2 + dt*dt*tmpOmegaMP*tmpOmegaMP / 4);
		dm0_y[i][j][k] = (1 + dt*tmpGammaM/2);
		dm1_y[i][j][k] = (-2);
		dm2_y[i][j][k] = (1 - dt*tmpGammaM/2);
	}
	std::cout << "Init Hy and By: rank " << rank << std::endl;

	//Hz
	for (int i = 0; i <= gridX - 1; ++i)
	for (int j = 0; j <= gridY - 1; ++j)
	for (int k = 0; k <= sliceSize - 2; ++k)
	{
		double posX = i + 1;
		double posY = j + 1;
		double posZ = addition + k + 1.5;

		double tmpMu = (Mu[i + 1][j + 1][k + 1] + Mu[i][j + 1][k + 1] +
			Mu[i + 1][j][k + 1] + Mu[i][j][k + 1]) / 4;

		DaHz[i][j][k] = (2 * EPS_Z * Ky(posY) - SigmaY(posX, posY) * dt) /
			(2 * EPS_Z * Ky(posY) + SigmaY(posX, posY) * dt);
		DbHz[i][j][k] = (2 * EPS_Z * Kz(posZ) + SigmaZ(posZ) * dt) /
			((2 * EPS_Z * Ky(posY) + SigmaY(posX, posY) * dt) * tmpMu * MU_Z);
		DcHz[i][j][k] = (2 * EPS_Z * Kz(posZ) - SigmaZ(posZ) * dt) /
			((2 * EPS_Z * Ky(posY) + SigmaY(posX, posY) * dt) * tmpMu * MU_Z);
	}

	//Bz
	for (int i = 0; i <= gridX - 2; ++i)
	for (int j = 0; j <= gridY - 2; ++j)
	for (int k = 0; k <= sliceSize - 1; ++k)
	{
		double posX = i + 1.5;
		double posY = j + 1.5;
		double posZ = addition + k + 1;

		DaBz[i][j][k] = (2 * EPS_Z * Kx(posX) - SigmaX(posX, posY) * dt) /
						(2 * EPS_Z * Kx(posX) + SigmaX(posX, posY) * dt);
		DbBz[i][j][k] = (2 * EPS_Z * dt) /
						((2 * EPS_Z * Kx(posX) + SigmaX(posX, posY) * dt));

		double tmpGammaM;
		double tmpOmegaMP;
		getGammaOmega(4, tmpOmegaMP, tmpGammaM, omegaMP[i + 1][j + 1][k + 1], omegaMP[i + 1][j + 1][k],
			gammaM[i + 1][j + 1][k + 1], gammaM[i + 1][j + 1][k]);

		C_z[i][j][k] = 1 + dt*tmpGammaM/2 + dt*dt*tmpOmegaMP*tmpOmegaMP/4;
		cm1_z[i][j][k] = (dt*dt*tmpOmegaMP*tmpOmegaMP/2 - 2);
		cm2_z[i][j][k] = (1 - dt*tmpGammaM/2 + dt*dt*tmpOmegaMP*tmpOmegaMP/4);
		dm0_z[i][j][k] = (1 + dt*tmpGammaM/2);
		dm1_z[i][j][k] = (-2);
		dm2_z[i][j][k] = (1 - dt*tmpGammaM/2);
	}
	std::cout << "Init Hz and Bz: rank " << rank << std::endl;

	if (rank != numProcs - 1)
	{
		//Dx
		for (int i = 0; i <= gridX - 1; ++i)
		for (int j = 0; j <= gridY - 2; ++j)
		{
			double posX = i + 1;
			double posY = j + 1.5;
			double posZ = addition + sliceSize + 0.5;

			CaDx_join[i][j] = (2 * EPS_Z * Ky(posY) - SigmaY(posX, posY) * dt) /
							  (2 * EPS_Z * Ky(posY) + SigmaY(posX, posY) * dt);
			CbDx_join[i][j] = (2 * EPS_Z * dt) /
							  ((2 * EPS_Z * Ky(posY) + SigmaY(posX, posY) * dt));

			double tmpGammaE;
			double tmpOmegaEP;
			
			getGammaOmega(2, tmpOmegaEP, tmpGammaE, omegaEP[i + 1][j + 1][sliceSize], omegaEP[i][j + 1][sliceSize],
				gammaE[i + 1][j + 1][sliceSize], gammaE[i][j + 1][sliceSize]);

			A_x_join_right[i][j] = 1 + dt*tmpGammaE / 2 + dt*dt*tmpOmegaEP*tmpOmegaEP / 4;
			am1_x_join_right[i][j] = (dt*dt*tmpOmegaEP*tmpOmegaEP / 2 - 2);
			am2_x_join_right[i][j] = (1 - dt*tmpGammaE / 2 + dt*dt*tmpOmegaEP*tmpOmegaEP / 4);
			bm0_x_join_right[i][j] = (1 + dt*tmpGammaE/2);
			bm1_x_join_right[i][j] = (-2);
			bm2_x_join_right[i][j] = (1 - dt*tmpGammaE/2);
		}
		std::cout << "Init Dx join: rank " << rank << std::endl;

		//Dy
		for (int i = 0; i <= gridX - 2; ++i)
		for (int j = 0; j <= gridY - 1; ++j)
		{
			double posX = i + 1.5;
			double posY = j + 1;
			double posZ = addition + sliceSize + 0.5;

			CaDy_join[i][j] = (2 * EPS_Z * Kz(posZ) - SigmaZ(posZ) * dt) /
							  (2 * EPS_Z * Kz(posZ) + SigmaZ(posZ) * dt);
			CbDy_join[i][j] = (2 * EPS_Z * dt) /
							  ((2 * EPS_Z * Kz(posZ) + SigmaZ(posZ) * dt));

			double tmpGammaE;
			double tmpOmegaEP;
			getGammaOmega(2, tmpOmegaEP, tmpGammaE, omegaEP[i + 1][j + 1][sliceSize], omegaEP[i + 1][j][sliceSize],
				gammaE[i + 1][j + 1][sliceSize], gammaE[i + 1][j][sliceSize]);

			A_y_join_right[i][j] = 1 + dt*tmpGammaE / 2 + dt*dt*tmpOmegaEP*tmpOmegaEP / 4;
			am1_y_join_right[i][j] = (dt*dt*tmpOmegaEP*tmpOmegaEP / 2 - 2);
			am2_y_join_right[i][j] = (1 - dt*tmpGammaE / 2 + dt*dt*tmpOmegaEP*tmpOmegaEP / 4);
			bm0_y_join_right[i][j] = (1 + dt*tmpGammaE/2);
			bm1_y_join_right[i][j] = (-2);
			bm2_y_join_right[i][j] = (1 - dt*tmpGammaE/2);
		}
		std::cout << "Init Dy join: rank " << rank << std::endl;

		//Bx
		for (int i = 0; i <= gridX - 1; ++i)
		for (int j = 0; j <= gridY - 2; ++j)
		{
			double posX = i + 1;
			double posY = j + 1.5;
			double posZ = addition + sliceSize + 0.5;

			DaBx_join[i][j] = (2 * EPS_Z * Ky(posY) - SigmaY(posX, posY) * dt) /
				(2 * EPS_Z * Ky(posY) + SigmaY(posX, posY) * dt);
			DbBx_join[i][j] = (2 * EPS_Z * dt) /
				((2 * EPS_Z * Ky(posY) + SigmaY(posX, posY) * dt));

			double tmpGammaM;
			double tmpOmegaMP;
			getGammaOmega(2, tmpOmegaMP, tmpGammaM, omegaMP[i + 1][j + 1][sliceSize], omegaMP[i][j + 1][sliceSize],
				gammaM[i + 1][j + 1][sliceSize], gammaM[i][j + 1][sliceSize]);

			C_x_join_right[i][j] = 1 + dt*tmpGammaM / 2 + dt*dt*tmpOmegaMP*tmpOmegaMP / 4;
			cm1_x_join_right[i][j] = (dt*dt*tmpOmegaMP*tmpOmegaMP / 2 - 2);
			cm2_x_join_right[i][j] = (1 - dt*tmpGammaM / 2 + dt*dt*tmpOmegaMP*tmpOmegaMP / 4);
			dm0_x_join_right[i][j] = (1 + dt*tmpGammaM / 2);
			dm1_x_join_right[i][j] = (-2);
			dm2_x_join_right[i][j] = (1 - dt*tmpGammaM / 2);
		}
		std::cout << "Init Bx join: rank " << rank << std::endl;

		//By
		for (int i = 0; i <= gridX - 2; ++i)
		for (int j = 0; j <= gridY - 1; ++j)
		{
			double posX = i + 1.5;
			double posY = j + 1;
			double posZ = addition + sliceSize + 0.5;

			DaBy_join[i][j] = (2 * EPS_Z * Kz(posZ) - SigmaZ(posZ) * dt) /
				(2 * EPS_Z * Kz(posZ) + SigmaZ(posZ) * dt);
			DbBy_join[i][j] = (2 * EPS_Z * dt) /
				((2 * EPS_Z * Kz(posZ) + SigmaZ(posZ) * dt));

			double tmpGammaM;
			double tmpOmegaMP;
			getGammaOmega(2, tmpOmegaMP, tmpGammaM, omegaMP[i + 1][j + 1][sliceSize], omegaMP[i + 1][j][sliceSize],
				gammaM[i + 1][j + 1][sliceSize], gammaM[i + 1][j][sliceSize]);

			C_y_join_right[i][j] = 1 + dt*tmpGammaM / 2 + dt*dt*tmpOmegaMP*tmpOmegaMP / 4;
			cm1_y_join_right[i][j] = (dt*dt*tmpOmegaMP*tmpOmegaMP / 2 - 2);
			cm2_y_join_right[i][j] = (1 - dt*tmpGammaM / 2 + dt*dt*tmpOmegaMP*tmpOmegaMP / 4);
			dm0_y_join_right[i][j] = (1 + dt*tmpGammaM / 2);
			dm1_y_join_right[i][j] = (-2);
			dm2_y_join_right[i][j] = (1 - dt*tmpGammaM / 2);
		}
		std::cout << "Init By join: rank " << rank << std::endl;

		//Ez
		for (int i = 0; i <= gridX - 1; ++i)
		for (int j = 0; j <= gridY - 1; ++j)
		{
			double posX = i + 1;
			double posY = j + 1;
			double posZ = addition + sliceSize + 0.5;

			double tmpEps = (Eps[i + 1][j + 1][sliceSize] + Eps[i][j + 1][sliceSize] +
				Eps[i + 1][j][sliceSize] + Eps[i][j][sliceSize]) / 4;

			CaEz_join[i][j] = (2 * EPS_Z * Ky(posY) - SigmaY(posX, posY) * dt) /
				(2 * EPS_Z * Ky(posY) + SigmaY(posX, posY) * dt);
			CbEz_join[i][j] = (2 * EPS_Z * Kz(posZ) + SigmaZ(posZ) * dt) /
				((2 * EPS_Z * Ky(posY) + SigmaY(posX, posY) * dt) * tmpEps * EPS_Z);
			CcEz_join[i][j] = (2 * EPS_Z * Kz(posZ) - SigmaZ(posZ) * dt) /
				((2 * EPS_Z * Ky(posY) + SigmaY(posX, posY) * dt) * tmpEps * EPS_Z);
		}
		std::cout << "Init Ez join: rank " << rank << std::endl;

		//Hz
		for (int i = 0; i <= gridX - 1; ++i)
		for (int j = 0; j <= gridY - 1; ++j)
		{
			double posX = i + 1;
			double posY = j + 1;
			double posZ = addition + sliceSize + 0.5;
			
			double tmpMu = (Mu[i + 1][j + 1][sliceSize] + Mu[i][j + 1][sliceSize] +
				Mu[i + 1][j][sliceSize] + Mu[i][j][sliceSize]) / 4;
			
			DaHz_join[i][j] = (2 * EPS_Z * Ky(posY) - SigmaY(posX, posY) * dt) /
							  (2 * EPS_Z * Ky(posY) + SigmaY(posX, posY) * dt);
			DbHz_join[i][j] = (2 * EPS_Z * Kz(posZ) + SigmaZ(posZ) * dt) /
							  ((2 * EPS_Z * Ky(posY) + SigmaY(posX, posY) * dt) * tmpMu * MU_Z);
			DcHz_join[i][j] = (2 * EPS_Z * Kz(posZ) - SigmaZ(posZ) * dt) /
							  ((2 * EPS_Z * Ky(posY) + SigmaY(posX, posY) * dt) * tmpMu * MU_Z);		
		}
		std::cout << "Init Hz join: rank " << rank << std::endl;
	}

}

void preSetup()
{
	allocateAll();
	
	if (rank == 0)
		std::cout << "Allocated" << std::endl;
	if (allocatedCount < 1024/4)
		std::cout << "Rank: " << rank << ". Allocated: " << allocatedCount <<
		" bytes" << std::endl;
	else if (allocatedCount < 1024 * 1024/4)
		std::cout << "Rank: " << rank << ". Allocated: " << allocatedCount / 1024 <<
		" Kb" << std::endl;
	else if (allocatedCount < 1024 * 1024 * 1024/4)
		std::cout << "Rank: " << rank << ". Allocated: " << allocatedCount / (1024 * 1024) <<
		" Mb" << std::endl;
	else if (allocatedCount < 1024 * 1024 * 1024 * 1024/4)
		std::cout << "Rank: " << rank << ". Allocated: " << allocatedCount / (1024 * 1024 * 1024) <<
		" Gb" << std::endl;

	initializeAll();
	std::cout << "Initialized: rank " << rank << std::endl;
}
void afterSetup()
{
	freeAll();
	
	allocatedCount *= 4;
	deallocatedCount *= 4;
	if (allocatedCount < 1024)
		std::cout << "Rank: " << rank << ". Allocated: " << allocatedCount << 
			" bytes. Freed: " << deallocatedCount << " bytes." << std::endl;
	else if (allocatedCount < 1024 * 1024)
		std::cout << "Rank: " << rank << ". Allocated: " << allocatedCount/1024 << 
			" Kb. Freed: " << deallocatedCount << " Kb." << std::endl;
	else if (allocatedCount < 1024 * 1024 * 1024)
		std::cout << "Rank: " << rank << ". Allocated: " << allocatedCount/(1024*1024) << 
			" Mb. Freed: " << deallocatedCount/(1024*1024) << " Mb." << std::endl;
	else if (allocatedCount < 1024 * 1024 * 1024 * 1024)
		std::cout << "Rank: " << rank << ". Allocated: " << allocatedCount/(1024*1024*1024) << 
			" Gb. Freed: " << deallocatedCount/(1024*1024*1024) << " Gb." << std::endl;
}

void Epart()
{
	//incident wave
	//UpdateIncE();

	//send-recv, update, check for successfull share
	//===========First==============================
	if (rank != numProcs - 1)
		RecvH();
	if (rank != 0)
		SendH();
	UpdateD();
	MPI_Barrier(MPI_COMM_WORLD);

	if (rank != numProcs - 1)
	{
		MPI_Wait(&requestH1, &status);
		MPI_Wait(&requestH2, &status);
	}
	if (rank != numProcs - 1)
		CopyH();

	UpdateD_shared();
	UpdateD1();
	UpdateD1_shared();
	
	//===========Second=============================
	if (rank != numProcs - 1)
		RecvDz();
	if (rank != 0)
		SendDz();
	UpdateE();
	MPI_Barrier(MPI_COMM_WORLD);

	if (rank != numProcs - 1)
	{
		MPI_Wait(&requestDz1, &status);	
		MPI_Wait(&requestDz1_prev, &status);
	}
	if (rank != numProcs - 1)
		CopyDz();
	
	//===========Third==============================
	if (rank != 0)
		RecvDxDy();
	if (rank != numProcs - 1)
		SendDxDy();
	UpdateE_shared_1();
	MPI_Barrier(MPI_COMM_WORLD);

	if (rank != 0)
	{
		MPI_Wait(&requestDx1, &status);
		MPI_Wait(&requestDy1, &status);
		MPI_Wait(&requestDx1_prev, &status);
		MPI_Wait(&requestDy1_prev, &status);
	}
	if (rank != 0)
		CopyDxDy();

	UpdateE_shared_2();
	ResetE();
	MPI_Barrier(MPI_COMM_WORLD);
}
void Hpart()
{
	//send-recv, update, check for successfull share
	//===========First==============================
	if (rank != numProcs - 1)
		RecvE();
	if (rank != 0)
		SendE();
	UpdateB();
	MPI_Barrier(MPI_COMM_WORLD);

	if (rank != numProcs - 1)
	{
		MPI_Wait(&requestE1, &status);
		MPI_Wait(&requestE2, &status);
	}
	if (rank != numProcs - 1)
		CopyE();

	UpdateB_shared();
	UpdateB1();
	UpdateB1_shared();

	//===========Second=============================
	if (rank != numProcs - 1)
		RecvBz();
	if (rank != 0)
		SendBz();
	UpdateH();
	MPI_Barrier(MPI_COMM_WORLD);

	if (rank != numProcs - 1)
	{
		MPI_Wait(&requestBz1, &status);
		MPI_Wait(&requestBz1_prev, &status);
	}
	if (rank != numProcs - 1)
		CopyBz();

	//===========Third==============================
	if (rank != 0)
		RecvBxBy();
	if (rank != numProcs - 1)
		SendBxBy();
	UpdateH_shared_1();
	MPI_Barrier(MPI_COMM_WORLD);

	if (rank != 0)
	{
		MPI_Wait(&requestBx1, &status);
		MPI_Wait(&requestBy1, &status);
		MPI_Wait(&requestBx1_prev, &status);
		MPI_Wait(&requestBy1_prev, &status);
	}
	if (rank != 0)
		CopyBxBy();

	UpdateH_shared_2();
	ResetH();
	MPI_Barrier(MPI_COMM_WORLD);
}

void updateAmplitude()
{
	/*for (int i = 0; i <= gridX - 1; i++)
	for (int j = 0; j <= gridY - 2; j++)
	for (int k = 0; k <= sliceSize - 2; k++)
	{
		if (std::fabs(Ex[i][j][k]) > Ex_amp[i][j][k])
			Ex_amp[i][j][k] = std::fabs(Ex[i][j][k]);
	}

	if (rank != numProcs - 1)
	{
		for (int i = 0; i <= gridX - 1; i++)
		for (int j = 0; j <= gridY - 2; j++)
		{
			if (std::fabs(Ex_join_right[i][j]) > Ex_join_right_amp[i][j])
				Ex_join_right_amp[i][j] = std::fabs(Ex_join_right[i][j]);
		}
	}*/

	for (int i = 0; i <= gridX - 2; i++)
	for (int j = 0; j <= gridY - 2; j++)
	for (int k = 0; k <= sliceSize - 1; k++)
	{
		if (std::fabs(Ez[i][j][k]) > Ez_amp[i][j][k])
			Ez_amp[i][j][k] = std::fabs(Ez[i][j][k]);
	}
}

double dipoleSource(double t, double x, double z, double l, int sign)
{
	double k0 = OMEGA / CC;
	double tmp = (1.0/1000000000) * sqrtf(2 / (PI*k0)) * expf(OMEGA*t + k0*sqrtf(x*x + (z + sign*l)*(z + sign*l)) - PI / 4) *  (z + sign*l) *
		(k0 - 1 / (2 * sqrtf(sqrtf(x*x + (z + sign*l)*(z + sign*l))))) / (x*x + (z + sign*l)*(z + sign*l));
	std::cout << tmp << " " << (x*x + (z + sign*l)*(z + sign*l)) << std::endl;
	return tmp;
}
//======================Main===============================
int main(int argc, char** argv)
{
	int i, j, k;
	double startTime;

	MPI_Init(&argc, &argv);
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	MPI_Comm_size(MPI_COMM_WORLD, &numProcs);

	calculateAmplitude = false;

	gridX = 128;
	gridY = 128;
	gridZ = 8;

	pmlL = 20;
	pmlR = 20;

	/*tfL = 40;
	tfR = 40;

	inc_size = gridX + 2;*/

	step = 0;
	double stepLambda = 20;
	numTimeSteps = 10;
	amplitudeSteps = 0;
	if (calculateAmplitude)
		amplitudeSteps = 2000;

	dx = LAMBDA / stepLambda;
	dy = LAMBDA / stepLambda;
	dz = LAMBDA / stepLambda;

	delta = dx;

	dt = dx / (2.0*CC);

	sliceSize = gridZ / numProcs;
	addSize = gridZ - sliceSize*numProcs;
	if (rank == numProcs - 1)
		sliceSize += addSize;

	printf("node:%d - %d \n.", rank, sliceSize);
	std::cout << "node: " << rank << "; size: " << gridX << " * " << 
		gridY << " * " << sliceSize << std::endl;

	preSetup();

	p0 = EPS_Z*1000/(OMEGA*dt);
	MPI_Barrier(MPI_COMM_WORLD);

	if (rank == 0) startTime = MPI_Wtime();

	
	while (step < numTimeSteps + amplitudeSteps)
	{
		if (rank == 0)
		{
			if (calculateAmplitude && step >= numTimeSteps)
				std::cout << "Calculating amplitude: step " << step << ". ";
			else
				std::cout << "Solving step " << step << " of " << numTimeSteps << ". ";
		}

		Epart();
		
		if (calculateAmplitude && step >= numTimeSteps)
		{
			/*updateAmplitude();

			if (rank == 0 && step < numTimeSteps + PERIOD/dt) 
			{
				for (int i = 0; i <= gridX - 2; ++i)
				for (int j = 0; j <= gridY - 2; ++j)
				{
					if ((i - gridX / 2 - 1) * (i - gridX / 2 - 1) + (j - gridY / 2 - 1) * (j - gridY / 2 - 1) < 41)
					{
						W += (dt * delta * delta * 4 * PI * 0.5 * OMEGA * Ez[i][j][sliceSize / 2] * Ez[i][j][sliceSize / 2] / OMEGA);
					}
				}
			}*/
		}

		if (rank == 0 && step * dt > PERIOD && step % 10 == 0)
		{
			/*Wstep[step/10] = 0.0;
			W1[step / 10] = 0.0;

			for (int i = 0; i <= gridX - 2; ++i)
			for (int j = 0; j <= gridY - 2; ++j)
			{
				if ((i - gridX / 2 - 1) * (i - gridX / 2 - 1) + (j - gridY / 2 - 1) * (j - gridY / 2 - 1) < 41)
				{
					Wstep[step / 10] += (OMEGA * 0.5 * Ez[i][j][sliceSize / 2] * Ez[i][j][sliceSize / 2] / (4 * PI));
				}
			}

			for (int step1 = step; step1 >= step - PERIOD / (dt); --step1)
			{
				W1[step / 10] += Wstep[step1 / 10] * dt;
			}
			W1[step / 10] /= PERIOD;
			W1[step / 10] /= (OMEGA * OMEGA * OMEGA * p0 * p0 * PI / (2 * CC * CC));

			std::cout << "| Wstep = " << Wstep[step / 10] << " W1 = " << W1[step / 10] << " |";*/
		}

		if (rank == 0)
		{
			timeAmp[step] = Hz[gridX/2][gridY/2][sliceSize / 2];
		}

		Hpart();

		if (rank == 0)
			std::cout << Ex[gridX / 2][gridY / 2][sliceSize / 2] << " " << Dx[gridX / 2][gridY / 2][sliceSize / 2] <<
				"Done." << std::endl;

		++step;

		if (rank == 0/* && step % 100 == 0*/)
		{
			//std::cout << "!!!!!" << Ez[gridX / 4][gridY / 2][sliceSize/2] << std::endl;
			{
				std::string buf("Dz[");
				std::ostringstream oss;
				oss << step;
				buf += oss.str();
				buf.append("].bmp");
				saveToBMP(Dz, gridX - 1, gridY - 1, sliceSize, "D:\\fdtd\\test", buf.c_str(), sliceSize / 2);

				std::string buf1("Dz[");
				std::ostringstream oss1;
				oss1 << step;
				buf1 += oss1.str();
				buf1.append("].txt");
				writeToFile(Dz, gridX - 1, gridY - 1, sliceSize, "D:\\fdtd\\test", buf1.c_str(), sliceSize / 2);
			}

			{
				std::string buf("Bx[");
				std::ostringstream oss;
				oss << step;
				buf += oss.str();
				buf.append("].bmp");
				saveToBMP(Bx, gridX, gridY - 1, sliceSize - 1, "D:\\fdtd\\test", buf.c_str(), sliceSize / 4);

				std::string buf1("Bx[");
				std::ostringstream oss1;
				oss1 << step;
				buf1 += oss1.str();
				buf1.append("].txt");
				writeToFile(Bx, gridX, gridY - 1, sliceSize - 1, "D:\\fdtd\\test", buf1.c_str(), sliceSize / 4);
			}

			{
				std::string buf("Ez[");
				std::ostringstream oss;
				oss << step;
				buf += oss.str();
				buf.append("].bmp");
				saveToBMP(Ez, gridX, gridY, sliceSize - 1, "D:\\fdtd\\test", buf.c_str(), sliceSize / 4);

				std::string buf1("Ez[");
				std::ostringstream oss1;
				oss1 << step;
				buf1 += oss1.str();
				buf1.append("].txt");
				writeToFile(Ez, gridX, gridY, sliceSize - 1, "D:\\fdtd\\test", buf1.c_str(), sliceSize / 4);
			}

			{
				std::string buf("By[");
				std::ostringstream oss;
				oss << step;
				buf += oss.str();
				buf.append("].bmp");
				saveToBMP(By, gridX - 1, gridY, sliceSize - 1, "D:\\fdtd\\test", buf.c_str(), sliceSize / 4);

				std::string buf1("By[");
				std::ostringstream oss1;
				oss1 << step;
				buf1 += oss1.str();
				buf1.append("].txt");
				writeToFile(By, gridX - 1, gridY, sliceSize - 1, "D:\\fdtd\\test", buf1.c_str(), sliceSize / 4);
			}

			{
				std::string buf("Hx[");
				std::ostringstream oss;
				oss << step;
				buf += oss.str();
				buf.append("].bmp");
				saveToBMP(Hx, gridX - 1, gridY, sliceSize, "D:\\fdtd\\test", buf.c_str(), sliceSize / 4);

				std::string buf1("Hx[");
				std::ostringstream oss1;
				oss1 << step;
				buf1 += oss1.str();
				buf1.append("].txt");
				writeToFile(Hx, gridX - 1, gridY, sliceSize, "D:\\fdtd\\test", buf1.c_str(), sliceSize / 4);
			}

			{
				std::string buf("Hy[");
				std::ostringstream oss;
				oss << step;
				buf += oss.str();
				buf.append("].bmp");
				saveToBMP(Hy, gridX, gridY - 1, sliceSize, "D:\\fdtd\\test", buf.c_str(), sliceSize / 4);

				std::string buf1("Hy[");
				std::ostringstream oss1;
				oss1 << step;
				buf1 += oss1.str();
				buf1.append("].txt");
				writeToFile(Hy, gridX, gridY - 1, sliceSize, "D:\\fdtd\\test", buf1.c_str(), sliceSize / 4);
			}
			/*std::string buf2("W[");
			std::ostringstream oss2;
			oss2 << step;
			buf2 += oss2.str();
			buf2.append("].bmp");
			saveToBMP(W1, step / 10, "D:\\fdtd\\test", buf2.c_str());

			std::string buf3("W[");
			std::ostringstream oss3;
			oss3 << step;
			buf3 += oss3.str();
			buf3.append("].txt");
			writeToFile(W1, step / 10, "D:\\fdtd\\test", buf3.c_str());*/
			//AllEz2D();
			/*if (rank == numProcs - 1)
			{
				std::cout << "!!!!!" << std::endl;
				std::string buf("Ez[");
				std::ostringstream oss;
				oss << step;
				buf += oss.str();
				buf.append("].bmp");
				saveToBMP(Ez[gridX / 2], gridY - 1, sliceSize, "D:\\fdtd", buf.c_str());
			}*/
		}
	}

	if (rank == 0)
	{
		//saveToBMP(timeAmp, numTimeSteps+amplitudeSteps, "D:\\fdtd\\test", "Hz_time.bmp");
		//W /= PERIOD;
		//std::cout << "W = " << W << std::endl;
	}
	if (calculateAmplitude)
	{
		Copy3darray(Ez, Ez_amp, gridX-1, gridY - 1, sliceSize);
		std::string buf("Ez[");
		std::ostringstream oss;
		oss << step;
		buf += oss.str();
		buf.append("].bmp");
		saveToBMP(Ez, gridX - 1, gridY - 1, sliceSize, "D:\\fdtd\\test", buf.c_str(), sliceSize / 2);
		std::string buf1("Ez[");
		std::ostringstream oss1;
		oss1 << step;
		buf1 += oss1.str();
		buf1.append("].txt");
		writeToFile(Ez, gridX - 1, gridY - 1, sliceSize, "D:\\fdtd\\test", buf1.c_str(), sliceSize / 2);
		//if (rank != numProcs - 1)
		//	Copy2darray(Ex_join_right, Ex_join_right_amp, gridX, gridY - 1);
	}
	//AllEz2D();

	/*//Ey
	for (int i = 0; i <= gridX - 2; ++i)
	for (int j = 0; j <= gridY - 1; ++j)
	for (int k = 0; k <= sliceSize - 2; ++k)
	{
		if (Ey[i][j][k] != 0.0)
			printf("Ey[%d][%d][%d] = %d\n", i, j, k, Ey[i][j][k]);
	}

	//Ez
	for (int i = 0; i <= gridX - 2; ++i)
	for (int j = 0; j <= gridY - 2; ++j)
	for (int k = 0; k <= sliceSize - 1; ++k)
	{
		if (Ez[i][j][k] != 0.0)
			printf("Ez[%d][%d][%d] = %d\n", i, j, k, Ez[i][j][k]);
	}

	//Hx
	for (int i = 0; i <= gridX - 2; ++i)
	for (int j = 0; j <= gridY - 1; ++j)
	for (int k = 0; k <= sliceSize - 1; ++k)
	{
		if (Hx[i][j][k] != 0.0)
			printf("Hx[%d][%d][%d] = %d\n", i, j, k, Hx[i][j][k]);
	}

	//Hy
	for (int i = 0; i <= gridX - 1; ++i)
	for (int j = 0; j <= gridY - 2; ++j)
	for (int k = 0; k <= sliceSize - 1; ++k)
	{
		if (Hy[i][j][k] != 0.0)
			printf("Hy[%d][%d][%d] = %d\n", i, j, k, Hy[i][j][k]);
	}

	//Hz
	for (int i = 0; i <= gridX - 1; ++i)
	for (int j = 0; j <= gridY - 1; ++j)
	for (int k = 0; k <= sliceSize - 2; ++k)
	{
		if (Hz[i][j][k] != 0.0)
			printf("Hz[%d][%d][%d] = %d\n", i, j, k, Hz[i][j][k]);
	}*/


	afterSetup();

#pragma omp parallel
	if (rank == 0)
		printf("Calc time:%lf\n Nproc = %d \n", MPI_Wtime() - startTime, numProcs);

	MPI_Finalize();
	return 0;
}
