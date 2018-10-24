#include <omp.h>
#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <gmp.h>

#define real float
#define NO_COLOR -1
#define DEBUG_COLOR -1

static int precision = 32;
static int maxiter = 30;
static int w = 800;
static int h = 600;
static mpf_t x_b;
static mpf_t y_b;
static mpf_t step;
static real *iterData = 0;
static int debug = 0;
static int gridsize = 10;

static real iterate_point(int x0_int, int y0_int)
{
	real val = iterData[y0_int * w + x0_int];
	if(val != NO_COLOR) {
		return val;
	}

	mpf_t x0;
	mpf_t y0;
	mpf_t valsq;
	mpf_t x;
	mpf_t y;
	mpf_t xt;
	mpf_t xs;
	mpf_t ys;
	mpf_t tmp;
	mpf_init2(x0, precision);
	mpf_init2(y0, precision);
	mpf_init2(valsq, precision);
	mpf_init2(x, precision);
	mpf_init2(y, precision);
	mpf_init2(xt, precision);
	mpf_init2(xs, precision);
	mpf_init2(ys, precision);
	mpf_init2(tmp, precision);

	mpf_mul_ui(x0, step, x0_int);
	mpf_add(x0, x0, x_b);
	mpf_mul_ui(y0, step, y0_int);
	mpf_add(y0, y0, y_b);

	int iter = 0;

	mpf_set(x, x0);
	mpf_set(y, y0);

	mpf_mul(xs, x, x);
	mpf_mul(ys, y, y);
	mpf_add(valsq, xs, ys);

	while((mpf_cmp_d(valsq, 4.0) < 0) && (iter < maxiter)) {
		mpf_sub(xt, xs, ys);
		mpf_add(xt, xt, x0);
		
		mpf_mul(tmp, x, y);
		mpf_mul_2exp(y, tmp, 1);
		mpf_add(y, y, y0);

		mpf_set(x, xt);

		mpf_mul(xs, x, x);
		mpf_mul(ys, y, y);
		
		mpf_add(valsq, xs, ys);

		++iter;
	}

	//double r_valsq = mpf_get_d(valsq);

	mpf_clear(x0);
	mpf_clear(y0);
	mpf_clear(valsq);
	mpf_clear(x);
	mpf_clear(y);
	mpf_clear(xt);
	mpf_clear(xs);
	mpf_clear(ys);
	mpf_clear(tmp);

	return (real)iter;
	//return (iter - log2(log2(r_valsq) * 0.5));
}

static int isNotEqualColor(real a, real b)
{
	return a != b;
}

static void fillRekt(int xb, int xe, int yb, int ye)
{
	int dx = xe-xb;
	int dy = ye-yb;
	//printf("%d %d\n", xb, dx);

	if((dy <= 1) || (dx <= 1)) {
		for(int i = yb; i <= ye; ++i) {
			for(int k = xb; k <= xe; ++k) {
				real val = iterData[i * w + k];
				if(val == NO_COLOR) {
					val = iterate_point(k, i);
					iterData[i * w + k] = val;
				}
			}
		}
		return;
	}

	real origVal = NO_COLOR;
	int same = 1;

	for(int i = xb; i <= xe; ++i) {
		real val = iterate_point(i, yb);
		iterData[yb * w + i] = val;

		if(origVal == NO_COLOR) {
			origVal = val;
		}
		else if(isNotEqualColor(origVal, val)) {
			same = 0;
		}
	}
	for(int i = xb; i <= xe; ++i) {
		real val = iterate_point(i, ye);
		iterData[ye * w + i] = val;

		if(origVal == NO_COLOR) {
			origVal = val;
		}
		else if(isNotEqualColor(origVal, val)) {
			same = 0;
		}
	}
	for(int i = yb; i <= ye; ++i) {
		real val = iterate_point(xb, i);
		iterData[i * w + xb] = val;

		if(origVal == NO_COLOR) {
			origVal = val;
		}
		else if(isNotEqualColor(origVal, val)) {
			same = 0;
		}
	}
	for(int i = yb; i <= ye; ++i) {
		real val = iterate_point(xe, i);
		iterData[i * w + xe] = val;

		if(origVal == NO_COLOR) {
			origVal = val;
		}
		else if(isNotEqualColor(origVal, val)) {
			same = 0;
		}
	}


	if(same) {
		for(int i = yb+1; (i <= (ye-1)) && (i < h); ++i) {
			for(int k = xb+1; (k <= (xe-1)) && (k < w); ++k) {
				if(debug) {
					iterData[i * w + k] = DEBUG_COLOR;
				}
				else {
					iterData[i * w + k] = origVal;
				}
			}
		}

		return;
	}

	if(dx > dy) {
		int midx = (xb + xe) / 2;
		fillRekt(xb, midx, yb, ye);
		fillRekt(midx, xe, yb, ye);
	}
	else {
		int midy = (yb + ye) / 2;
		fillRekt(xb, xe, yb, midy);
		fillRekt(xb, xe, midy, ye);
	}
}

static int min(int a, int b)
{
	return a < b ? a : b;
}

int main(int argc, char **argv)
{
	precision = 32;
	maxiter = 100;
	w = 1920;
	h = 1080;

	debug = 1;
	if(argc > 1) {
		debug = atoi(argv[1]);
	}

	gridsize = 16;

	mpf_init2(step, precision);
	mpf_init2(x_b, precision);
	mpf_init2(y_b, precision);

	mpf_set_d(step, 4.0 / w);
	mpf_set_d(x_b, -2.5);
	mpf_set_d(y_b, -1.125);

	iterData = malloc(sizeof(real) * w * h);
	for(int i = 0; i < w*h; ++i) {
		iterData[i] = NO_COLOR;
	}

/*
	//#pragma omp parallel for schedule(dynamic, 1)
	for(int wy = 0; wy < h; ++wy) {
		for(int wx = 0; wx < w; ++wx) {
			real iter = iterate_point(wx, wy);
			iterData[wy * w + wx] = iter;
		}
	}
*/

	//fillRekt(0, w/2, 0, h-1);
	//fillRekt(w/2, w-1, 0, h-1);

	const int istep = w/gridsize;
	const int hdirec = h/istep + 1;
	const int all = hdirec * gridsize;

	#pragma omp parallel for schedule(dynamic, 1)
	for(int i = 0; i < all; ++i) {
		int cubex = i % gridsize;
		int cubey = i / gridsize;

		int beginx = cubex * istep;
		int beginy = cubey * istep;

		int endx = min(beginx + istep, w-1);
		int endy = min(beginy + istep, h-1);

		fillRekt(beginx, endx, beginy, endy);
	}

	//test output in *.ppm
	printf("P3\n%d\n%d\n%d\n", w, h, 255);
	for(int wy = 0; wy < h; ++wy) {
		for(int wx = 0; wx < w; ++wx) {
			real val = iterData[wy * w + wx];
			if(val == DEBUG_COLOR) {
				printf("%d %d %d\n", 0, 128, 64);
				continue;
			}

			int colVal = (int)((1.0 - (val/maxiter)) * 255.0);
			if(colVal > 255) colVal = 255;
			if(colVal < 0) colVal = 0;
			printf("%d %d %d\n", colVal, colVal, colVal);
		}
	}

	mpf_clear(x_b);
	mpf_clear(y_b);
	mpf_clear(step);
	free(iterData);

	return 0;
}
