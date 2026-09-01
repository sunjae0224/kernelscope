NVCC ?= /usr/local/cuda/bin/nvcc
ARCH ?= sm_80

naive_attn: naive_attn.cu
	$(NVCC) -O3 -arch=$(ARCH) -lineinfo -o $@ $<

clean:
	rm -f naive_attn

.PHONY: clean
