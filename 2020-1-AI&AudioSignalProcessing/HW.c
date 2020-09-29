#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#define N			1600  
#define SAMPLE_RATE 16000
#define PI 			3.14

float comp_freq(int index);
float* Hanning_window(int win_size);
float* Hamming_window(int win_size);
float* windowing(float* signal, float* window, int win_size);
float convert_hertz(int k, int sr);
int DFT(float signal[], float spec_real[], float spec_imag[], float spec_magn[]);
int peak_magnitude(float spec_magn[]);

FILE* fin;


int skip[8] = {
	SAMPLE_RATE * 0.3, //start 0.3s
	SAMPLE_RATE * 0.4, //(# of sample per s)*[(0.8-0.4)s]
	SAMPLE_RATE * 0.22,//(# of sample per s)*[(1.12-0.9)s]
	SAMPLE_RATE * 0.12,//(# of sample per s)*[(1.4-1.22)s]
	SAMPLE_RATE * 0.1,//(# of sample per s)*[(1.6-1.5)s]
	SAMPLE_RATE * 0.1,//(# of sample per s)*[(1.8-1.7)s]
	SAMPLE_RATE * 0.2,//(# of sample per s)*[(2.1-1.9)s]
	SAMPLE_RATE * 0.2//(# of sample per s)*[(2.4-2.2)s]
};


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

int DFT(float signal[], float spec_real[], float spec_imag[], float spec_magn[]) {
	float* window = NULL;
	float* windowed = NULL;

	//window = Hanning_window(N);
	window = Hamming_window(N);
	windowed = windowing(signal, window, N);

	for (int k = 0; k < N; k++) {
		spec_real[k] = 0.0;
		spec_imag[k] = 0.0;

		for (int n = 0; n < N; n++) {
			spec_real[k] += signal[n] * cos(2 * PI * k * n / N);
			spec_imag[k] -= signal[n] * sin(2 * PI * k * n / N);
		}
		spec_magn[k] = pow(spec_real[k], 2) + pow(spec_imag[k], 2);
		spec_magn[k] = sqrt(spec_magn[k]);
	}

	free(window);
	return 0;
}

int peak_magnitude(float spec_magn[]) {
	float max_magnitued = 0.0;
	int argmax = 0;

	for (int k = 0; k < N; k++) {
		if (max_magnitued < spec_magn[k]) {
			max_magnitued = spec_magn[k];
			argmax = k;
		}
	}

	return argmax;
}


float convert_hertz(int k, int sr) {
	float freq = (float)sr / (float)N;

	return (float)k * freq;
}


float comp_freq(int i) { //Baseline
	float signal[N];
	float spec_real[N], spec_imag[N], spec_magn[N];
	float freq = 0.0;
	short data = 0;
	int k = 0;

	//Skip to the start
	for (int n = 0; n < skip[i]; n++)
		fread(&data, 2, 1, fin);

	for (int n = 0; n < N; n++) {
		fread(&data, 2, 1, fin);
		signal[n] = (float)data;
	}

	// window -> N-point DFT -> peak magnitude -> freq(Hz)

	DFT(signal, spec_real, spec_imag, spec_magn);
	k = peak_magnitude(spec_magn);
	freq = convert_hertz(k, SAMPLE_RATE);

	return freq;
}


int main() { //Baseline
	float note_freq[8] = { 0.0, };

	fopen_s(&fin, "input16k.raw", "rb");

	for (int i = 0; i < 8; i++) {
		note_freq[i] = comp_freq(i);
		printf("%d Note Freqency : %.2f \n", i+1, note_freq[i]);
	}

	fclose(fin);
	return 0;
}
