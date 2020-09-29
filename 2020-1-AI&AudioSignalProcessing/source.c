
#include <stdio.h>
#include <stdlib.h>
#include <math.h>



FILE* fin;
FILE* fout;


#define N			320  
#define SAMPLE_RATE 8000
#define PI 			3.14
#define ORDER       10


float* Hanning_window(int win_size);
float* Hamming_window(int win_size);
float* windowing(float* signal, float* window, int win_size);
float* DFT(float* signal, int n_dft);
float* get_R(float* signal, int sample_num, int order);
void Durbin(float* corrs, float* errors, float* reflect_coef, float** filter_coef, int stride);
float* mag_inverse(float* signal, int length);
float* zero_padding(int size);
float* get_an(float** filter_coef, int n_dft, int stride);



float* Hanning_window(int win_size) {
    float* window = (float*)malloc(sizeof(float) * win_size);

    for (int n = 0; n < win_size; n++)
        window[n] = 0.5 - (0.5 * cos(2 * PI * n / (float)(win_size - 1.0f)));

    return window;
}



float* Hamming_window(int win_size) {
    float* window = (float*)malloc(sizeof(float) * win_size);

    for (int n = 0; n < win_size; n++)
        window[n] = 0.54 - 0.46 * cos(2 * PI * n / (float)(win_size - 1.0f));

    return window;
}



float* windowing(float* signal, float* window, int win_size) {
    for (int n = 0; n < win_size; n++)
        signal[n] *= window[n];

}
float* DFT(float* signal, int n_dft) {
    float* spec_magn = zero_padding(n_dft);
    float* spec_real = zero_padding(n_dft);
    float* spec_imag = zero_padding(n_dft);

    for (int k = 0; k < n_dft; k++) {
        spec_real[k] = 0.0;
        spec_imag[k] = 0.0;

        for (int n = 0; n < n_dft; n++) {
            spec_real[k] += signal[n] * cos(2 * PI * k * n / (float)n_dft);
            spec_imag[k] -= signal[n] * sin(2 * PI * k * n / (float)n_dft);
        }
        spec_magn[k] = pow(spec_real[k], 2) + pow(spec_imag[k], 2);
        spec_magn[k] = sqrt(spec_magn[k]);
    }

    free(spec_real);
    free(spec_imag);
    return spec_magn;
}

float* masking(float* signal, int mask, int start, int end) {
    for (int i = start; i < end; i++) {
        signal[i] = mask;
    }

    return signal;
}


float* get_R(float* signal, int sample_num, int order) {
    float* corr = (float*)malloc(sizeof(float) * order);


    for (int k = 0; k <= order; k++) { //0~10
        float sum_s = 0;
        for (int n = 0; n < N; n++) {
            sum_s += signal[n] * signal[n - k];
        }
        corr[k] = sum_s;
    }

    return corr;
}

void Durbin(float* corrs, float* errors, float* reflect_coef, float** filter_coef, int stride) {
    float sum;
    errors[0] = corrs[0];

    for (int i = 1; i <= stride; i++) {
        sum = 0;
        for (int j = 1; j <= i - 1; j++) {
            sum += filter_coef[i - 1][j] * corrs[i - j];
        }

        reflect_coef[i] = (corrs[i] - sum) / errors[i - 1];
        filter_coef[i][i] = reflect_coef[i];

        for (int j = 1; j <= i - 1; j++) {
            filter_coef[i][j] = filter_coef[i - 1][j] - reflect_coef[i] * filter_coef[i - 1][i - j];
        }
        errors[i] = (1 - pow(reflect_coef[i], 2)) * errors[i - 1];
    }

    return;
}


void Durbin(float* corrs, float* errors, float* reflect_coef, float** filter_coef, int stride) {
    float sum;
    errors[0] = corrs[0];

    for (int i = 1; i <= stride; i++) {
        sum = 0;
        for (int j = 1; j <= i - 1; j++) {
            sum += filter_coef[i - 1][j] * corrs[i - j];
        }

        reflect_coef[i] = (corrs[i] - sum) / errors[i - 1];
        filter_coef[i][i] = reflect_coef[i];

        for (int j = 1; j <= i - 1; j++) {
            filter_coef[i][j] = filter_coef[i - 1][j] - reflect_coef[i] * filter_coef[i - 1][i - j];
        }
        errors[i] = (1 - pow(reflect_coef[i], 2)) * errors[i - 1];
    }

    return;
}


float* get_an(float** filter_coef, int n, int order) {
    float* transfer_func = zero_padding(n);
    transfer_func[0] = 1;

    for (int i = 1; i < n; i++) {
        if (i < order)
            transfer_func[i] = -filter_coef[order][i + 1];

        else
            transfer_func[i] = 0;
    }

    return transfer_func;
}


float* zero_padding(int size) {
    float* zeros = (float*)malloc(sizeof(float) * size);

    for (int i = 0; i < size; i++)
        zeros[i] = 0.0;

    return zeros;
}


float* mag_inverse(float* signal, int length) {
    float* mag_inversed = zero_padding(length);

    for (int i = 0; i < length; i++)
        mag_inversed[i] = 1 / signal[i];

    return mag_inversed;
}


int main(void) {
    FILE* fin, * fnc, * fde;

    float* signal = (float*)malloc(sizeof(float) * N);
    float* window = (float*)malloc(sizeof(float) * N);
    float* errors = zero_padding(ORDER + 1);
    float* reflect_coef = zero_padding(ORDER + 1);
    float** filter_coef = (double**)malloc(sizeof(double*) * (ORDER + 1));
    float* corrs, * transfer_func, * envelope, * spec_magn;
    short data[ORDER];


    fopen_s(&fin, "Male.raw", "rb");
    fopen_s(&fout, "spectral_envelop.txt", "wb");


    for (int i = 0; i < N; i++) {
        fread(data, 2, 1, fin);
        signal[i] = (double)*data;

    }

    window = Hanning_window(N);

    //window = Hamming_window( N);
    signal = windowing(signal, window, N);
    signal = masking(signal, 0, 160, N);
    corrs = get_R(signal, N, ORDER);
    Durbin(corrs, errors, reflect_coef, filter_coef, ORDER);

    transfer_func = get_an(filter_coef, N, ORDER);
    spec_magn = DFT(transfer_func, N);
    envelope = mag_inverse(spec_magn, N);

    for (int i = 0; i < N; i++)
        fprintf(fout, "%10.7f\n", envelope[i]);

    fclose(fin);
    fclose(fout);


    return 0;
}