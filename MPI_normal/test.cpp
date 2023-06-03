//#include<mpi.h>
//#include<iostream>
//#include<Windows.h>
//using namespace std;
//int N;
//float gdata[10000][10000];
//
//void Initialize(int N,float temp[][10000])
//{
//	for (int i = 0; i < N; i++)
//	{
//		//���Ƚ�ȫ��Ԫ����Ϊ0���Խ���Ԫ����Ϊ1
//		for (int j = 0; j < N; j++)
//		{
//			temp[i][j] = 0;
//		}
//		temp[i][i] = 1.0;
//		//�������ǵ�λ�ó�ʼ��Ϊ�����
//		for (int j = i + 1; j < N; j++)
//		{
//			temp[i][j] = (float)rand();
//		}
//	}
//	for (int k = 0; k < N; k++)
//	{
//		for (int i = k + 1; i < N; i++)
//		{
//			for (int j = 0; j < N; j++)
//			{
//				temp[i][j] += temp[k][j];
//			}
//		}
//	}
//}
////�����㷨
//void Normal(int N,float gdata1[][10000])
//{
//	int i, j, k;
//	for (k = 0; k < N; k++)
//	{
//		for (j = k + 1; j < N; j++)
//		{
//			gdata1[k][j] = gdata1[k][j] / gdata1[k][k];
//		}
//		gdata1[k][k] = 1.0;
//		for (i = k + 1; i < N; i++)
//		{
//			for (j = k + 1; j < N; j++)
//			{
//				gdata1[i][j] = gdata1[i][j] - (gdata1[i][k] * gdata1[k][j]);
//			}
//			gdata1[i][k] = 0;
//		}
//	}
//}
////mpi�л���,������
//double MPI_row(float gdata[][10000])
//{
//	int rank, size;//�������͵�ǰ���̱��
//	double start_time, end_time;
//	MPI_Init(NULL, NULL);
//	MPI_Comm_size(MPI_COMM_WORLD, &size);
//	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
//	//Ϊ����mpi���ԣ�0�Ž��̽��г�ʼ��
//	if (rank == 0)
//		/*Initialize(N, gdata);*/
//
//	MPI_Barrier(MPI_COMM_WORLD);//ͬ�����ټ�ʱ����������ʱҲ�����
//
//
//	start_time = MPI_Wtime();//��0�Ž��̼�¼ʱ��
//
//	//�������ÿ�����̿�ʼ�ͽ�������Ļ���
//	int part = N / size;
//	int remainder = N % size;
//	int start = rank * part + (rank < remainder ? rank : remainder);
//	int end = start + part + (rank < remainder ? 1 : 0);
//	if (rank == size - 1) {
//		end = N;
//	}
//	//0�Ž��̸�������Ϣ��������̸��������Ϣ
//	if (rank == 0)
//	{
//		for (int i = 1; i < size; i++)
//		{
//			int temp_start = i * part + (i < remainder ? i : remainder);
//			int temp_end = temp_start + part + (i < remainder ? 1 : 0);
//			if (i == size - 1) {
//				temp_end = N;
//			}
//			MPI_Send(&gdata[temp_start][0], N * (temp_end - temp_start + 1), MPI_FLOAT, i, 0, MPI_COMM_WORLD);
//		}
//	}
//	else {
//		MPI_Recv(&gdata[start][0], N * (end - start + 1), MPI_FLOAT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
//	}
//	//��Ԫ���㲿�֣�����0�Ž��������еȴ���ֱ࣬����0�Ž��̽��г�������
//	//for (int k = 0; k < N; k++) {
//	//	if (rank == 0)
//	//	{
//	//		for (int j = k + 1; j < N; j++)
//	//		{
//	//			gdata[k][j] /= gdata[k][k];
//	//		}
//	//		gdata[k][k] = 1.0;
//	//		for (int j = 1; j < size; j++)//�㲥����������
//	//		{
//	//			//0�Ž��̷��͸�����������Ϣ
//	//			MPI_Send(&gdata[k][0], N, MPI_FLOAT, j, 0, MPI_COMM_WORLD);
//	//		}
//	//	}
//	for(int k=0;k<N;k++)
//	{
//		if (k >= start && k <= end)
//		{
//			for (int j = k + 1; j < N; j++)
//				{
//					gdata[k][j] /= gdata[k][k];
//				}
//			gdata[k][k] = 1.0;
//			for (int p = 0; p < size; p++) {
//				if (p != rank)
//					MPI_Send(&gdata[k][0], N, MPI_FLOAT, p, 0, MPI_COMM_WORLD);
//			}
//		}
//		else {
//			//�������̽�����Ϣ
//			MPI_Recv(&gdata[k][0], N, MPI_FLOAT, MPI_ANY_SOURCE, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
//		}
//		//��Ԫ����
//		for (int i =max(k+1,start); i <= end; i++)
//		{
//			for (int j = k + 1; j < N; j++)
//			{
//				gdata[i][j] = gdata[i][j] - gdata[i][k] * gdata[k][j];
//			}
//			gdata[i][k] = 0;
//		}
//	}
//	MPI_Barrier(MPI_COMM_WORLD);
//		end_time = MPI_Wtime();
//	MPI_Finalize();
//	return(end_time - start_time) * 1000;
//}
//
//int main() {
//	cin >> N;
//	LARGE_INTEGER fre, begin, end;
//	double gettime;
//
//
//
//
//	QueryPerformanceFrequency(&fre);
//	QueryPerformanceCounter(&begin);
//	//Initialize(N, gdata1);
//	QueryPerformanceCounter(&end);
//	gettime = (double)((end.QuadPart - begin.QuadPart) * 1000.0) / (double)fre.QuadPart;
//	cout << "intial time: " << gettime << " ms" << endl;
//
//	QueryPerformanceFrequency(&fre);
//	QueryPerformanceCounter(&begin);
//	//Normal(N, gdata1);
//	QueryPerformanceCounter(&end);
//	gettime = (double)((end.QuadPart - begin.QuadPart) * 1000.0) / (double)fre.QuadPart;
//	cout << "normal time: " << gettime << " ms" << endl;
//
//	cout << "MPI time: " << MPI_row(gdata) << " ms" << endl;
//
//}