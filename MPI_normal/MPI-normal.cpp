#include <iostream>
#include <mpi.h>
#include<Windows.h>
#include <omp.h>
#include<cstring>
using namespace std;

float matrix[10000][10000];
float matrix1[10000][10000];
float matrix2[10000][10000];
float matrix3[10000][10000];
void Initialize(int N)
{
    for (int i = 0; i < N; i++)
    {
        //���Ƚ�ȫ��Ԫ����Ϊ0���Խ���Ԫ����Ϊ1
        for (int j = 0; j < N; j++)
        {
            matrix[i][j] = 0;
            matrix1[i][j] = 0;
            matrix2[i][j] = 0;
            matrix3[i][j] = 0;
        }
        matrix[i][i] = matrix1[i][i] = matrix2[i][i] = matrix3[i][i] = 1.0;
        //�������ǵ�λ�ó�ʼ��Ϊ�����
        for (int j = i + 1; j < N; j++)
        {
            matrix[i][j] = matrix1[i][j] = matrix2[i][j] = matrix3[i][j] = rand();
        }
    }
    for (int k = 0; k < N; k++)
    {
        for (int i = k + 1; i < N; i++)
        {
            for (int j = 0; j < N; j++)
            {
                matrix[i][j] += matrix[k][j];
                matrix1[i][j] += matrix1[k][j];
                matrix2[i][j] += matrix2[k][j];
                matrix3[i][j] += matrix3[k][j];

            }
        }
    }
}
//ƽ���㷨
void normal(float matrix[][10000], int N) {
    for (int k = 0; k < N; k++) {
        for (int j = k + 1; j < N; j++)
            matrix[k][j] /= matrix[k][k];
        matrix[k][k] = 1.0;
        for (int i = k + 1; i < N; i++) {
            for (int j = k + 1; j < N; j++)
                matrix[i][j] = matrix[i][j] - matrix[i][k] * matrix[k][j];
            matrix[i][k] = 0;
        }
    }
}
//һά�п黮��
void MPI_row(float matrix[][10000], int N) {
    int rank;
    int size;
    double begin_time, end_time;
    
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    if (rank == 0)
        begin_time = MPI_Wtime();
    else
        begin_time = 0;
    int part = N / size;//ÿ�����̷��������
    int start = rank * part ;
    int end = start + part-1 ;
    if (rank == size - 1) {
        end = N-1;
        part = N - start;//���һ�����̷��������
    }
    // ����Ԫ����
    for (int k = 0; k < N; k++) 
    {
        // �鿴�Ƿ��ɱ����̸����������
        if (k >= start && k <= end)
        {
            for (int j = k + 1; j < N; j++)
                matrix[k][j] /= matrix[k][k];
            matrix[k][k] = 1;
            for (int p = 0; p < size; p++) {
                if (p != rank)
                    MPI_Send(&matrix[k][0], N, MPI_FLOAT, p, 0, MPI_COMM_WORLD);//�㲥����������
            }
        }
        // ������̽���
        else {
            MPI_Recv(&matrix[k][0], N, MPI_FLOAT, MPI_ANY_SOURCE, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        }
        // ������Ԫ����,��ĳ�еķ�Χ�ڲŸ���
        for (int i = max(k + 1,start); i <= end; i++)
        {
            for (int j = k + 1; j < N; j++) {
                matrix[i][j] = matrix[i][j] - matrix[i][k] * matrix[k][j];
            }
            matrix[i][k] = 0;
        }
    }
    MPI_Barrier(MPI_COMM_WORLD);
    //��ȥ��ɣ����лش�����,����ش���0�Ž���
    if (rank != 0)
        MPI_Send(&matrix[start][0], part*N, MPI_FLOAT, 0, 0, MPI_COMM_WORLD);
    //{
    //    for(int i=start;i<=end;i++)
    //        MPI_Send(&matrix[i][0],  N, MPI_FLOAT, 0, 0, MPI_COMM_WORLD);
    //}
    else//���������̽��ս�������
        for(int i=1;i<size;i++)
            MPI_Recv(&matrix[end + 1+(i-1)*part][0], part * N, MPI_FLOAT, i, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

    end_time = MPI_Wtime();
    if (rank == 0)
        cout << "MPI_row time: " << (end_time -begin_time) * 1000 << " ms" << endl;
 
}
//���ù㲥���ͷַ����շ�����Ϣ
void MPI_broadcast(float matrix[][10000], int N) {
    int rank;
    int size;
    double begin_time, end_time;

    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    if (rank == 0)
        begin_time = MPI_Wtime();
    else
        begin_time = 0;
    int part = N / size;//ÿ�����̷��������
    int start = rank * part;
    int end = start + part - 1;
    if (rank == size - 1) {
        end = N - 1;
        part = N - start;//���һ�����̷��������
    }
    // ����Ԫ����
    for (int k = 0; k < N; k++)
    {
        // ͳһ��0�Ž��̽��г����������Ӵ�������
        if (rank == 0)
        {
            for (int j = k + 1; j < N; j++)
                matrix[k][j] /= matrix[k][k];
            matrix[k][k] = 1;
            //���߳̾͵ز���
            MPI_Scatter(&matrix[k][0], N, MPI_FLOAT,MPI_IN_PLACE , N, MPI_FLOAT, 0, MPI_COMM_WORLD);
        }
        else
            //�����߳�
            MPI_Scatter(&matrix[k][0], N, MPI_FLOAT, &matrix[k][0], N, MPI_FLOAT, 0, MPI_COMM_WORLD);
        // ������Ԫ����,��ĳ�еķ�Χ�ڲŸ���
        for (int i = max(k + 1, start); i <= end; i++)
        {
            for (int j = k + 1; j < N; j++) {
                matrix[i][j] = matrix[i][j] - matrix[i][k] * matrix[k][j];
            }
            matrix[i][k] = 0;
        }
    }
    MPI_Barrier(MPI_COMM_WORLD);
    //��ȥ��ɣ����лش�����,����ش���0�Ž���
    if(rank!=0)
        MPI_Gather(&matrix[start][0], N * part, MPI_FLOAT, &matrix[start][0], N * part, MPI_FLOAT, 0, MPI_COMM_WORLD);
    else
        MPI_Gather(MPI_IN_PLACE, N * part, MPI_FLOAT, &matrix[0][0], N * part, MPI_FLOAT, 0, MPI_COMM_WORLD);
    end_time = MPI_Wtime();
    if (rank == 0)
        cout << "MPI_gather time: " << (end_time - begin_time) * 1000 << " ms" << endl;

}

//�л��֣��Զ�����������
void MPI_col(float matrix[][10000], int N) {
    int rank;
    int size;
    double begin_time, end_time;

    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    if (rank == 0)
        begin_time = MPI_Wtime();
    else
        begin_time = 0;
    MPI_Datatype matrix_col;//�����е���������
    MPI_Type_vector(N, 1, N, MPI_FLOAT, &matrix_col);//���ݽṹ������
    //�ύ
    MPI_Type_commit(&matrix_col);
    int part = N / size;//ÿ�����̷��������
    int start = rank * part;
    int end = start + part - 1;
    if (rank == size - 1) {
        end = N - 1;
        part = N - start;//���һ�����̷��������
    }
    // ����Ԫ����
    MPI_Request* request=new MPI_Request[size];
    for (int k = 0; k < N; k++)
    {
        // �鿴�Ƿ��ɱ����̸���������㣬������
        if (k >= start && k <= end)
        {
            //�ȷ�����ͨ�Ź㲥���ټ���
            for (int p = 0; p < size; p++) {
                if (p != rank)
                    MPI_Send(&matrix[0][k], 1, matrix_col, p, 0, MPI_COMM_WORLD);//��һ�й㲥����������
            }
            matrix[k][k] = 1;
            for (int i = k + 1; i < N; i++)
                matrix[k][i] = 0;

        }
        // ������̽���
        else {
            MPI_Recv(&matrix[0][k], 1, matrix_col, MPI_ANY_SOURCE, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        }
        // ������̽�����Ԫ����,ֻ�����������
            for (int i = max(k+1,start); i <=end; i++)
            {
                matrix[i][k] /= matrix[k][k];
                    for (int j = k + 1; j < N; j++) {
                        matrix[j][i] = matrix[j][i] - matrix[i][k] * matrix[j][k];
                    }
            }
    }
    MPI_Barrier(MPI_COMM_WORLD);
    ////��ȥ��ɣ����лش�����,����ش���0�Ž���
    //if (rank != 0)
    //    MPI_Send(&matrix[0][start], part, matrix_col, 0, 0, MPI_COMM_WORLD);

    //else//���������̽��ս�������
    //    for (int i = 1; i < size; i++)
    //        MPI_Recv(&matrix[0][end + 1 + (i - 1) * part], part, matrix_col, i, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

    end_time = MPI_Wtime();
    //�ͷ�
    MPI_Type_free(&matrix_col);
    if (rank == 0)
        cout << "MPI_col time: " << (end_time - begin_time) * 1000 << " ms" << endl;

}

int main() {


    int N =3000;
	LARGE_INTEGER fre, begin, end;
	double gettime;
    int rank;
    MPI_Init(NULL, NULL);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	QueryPerformanceFrequency(&fre);
	QueryPerformanceCounter(&begin);
    Initialize(N);
	QueryPerformanceCounter(&end);
	gettime = (double)((end.QuadPart - begin.QuadPart) * 1000.0) / (double)fre.QuadPart;
    MPI_Barrier(MPI_COMM_WORLD);
    if (rank == 0)
        cout << "intial time: " << gettime << " ms" << endl;
    
    
    QueryPerformanceFrequency(&fre);
    QueryPerformanceCounter(&begin);
    normal(matrix, N);
    QueryPerformanceCounter(&end);
    gettime = (double)((end.QuadPart - begin.QuadPart) * 1000.0) / (double)fre.QuadPart;
    MPI_Barrier(MPI_COMM_WORLD);
    if(rank==0)
        cout << "normal time: " << gettime << " ms" << endl;


    MPI_row(matrix1, N);
    MPI_broadcast(matrix2, N);
    MPI_col(matrix3, N);
    MPI_Finalize();
    
}