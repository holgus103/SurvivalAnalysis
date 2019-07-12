# nur Schweiz
import math
# calculates the correlations between variables and returns a symetrix matrix in form of a list of lists

def calculate_correlation_matrix(vars):
    avgs = [];

    for var in vars:
       avgs.append(sum(var)/len(var));

    res = [];

    for i in range(0,len(vars)):
        res_row = [];
        for j in range(0, i):
            if i == j:
                continue;
            else:
                sum_up = 0;
                sum_sqr_x = 0;
                sum_sqr_y = 0;
                for k in range(0, len(vars[0])):
                    mul_x = vars[i][k] - avgs[i];
                    mul_y = vars[j][k] - avgs[j];
                    sum_up += mul_x * mul_y;
                    sum_sqr_x += mul_x * mul_x;
                    sum_sqr_y += mul_y * mul_y;
                res_row.append(sum_up/(math.sqrt(sum_sqr_x) * math.sqrt(sum_sqr_y)));

        res.append(res_row);
    return res;

# function for pretty printing of the triagonal format of the matrix as its symetrical
def pretty_print(matrix):
    for j in range(0, len(matrix) + 1):
        for i in range(0, j):
            print('{0}, '.format(matrix[j-1][i]), end='');
        print('1, ', end = '');
        for i in range(j, len(matrix)-1):
            print('{0}, '.format(matrix[i][j]), end='');
        if(j < len(matrix)):
            print('{0} '.format(matrix[len(matrix)-1][j]));
        else:
            print(' ');

def generate_full_matrix(matrix_triangular):
    matrix = [];
    for j in range(0, len(matrix_triangular) + 1):
        row = []
        for i in range(0, j):
            row.append(matrix_triangular[j-1][i]);
        row.append(1);
        for i in range(j, len(matrix_triangular)-1):
            row.append(matrix_triangular[i][j]);
        if(j < len(matrix_triangular)):
            row.append(matrix_triangular[len(matrix_triangular)-1][j]);
        matrix.append(row);
    return matrix;
