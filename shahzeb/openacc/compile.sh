#!/bin/bash
gcc -o sgemvcpu sgemv_cpu.c -O3 -lm
pgcc -ta=nvidia,time -acc -Minfo=accel sgemv1.c -o sgemv1 -O3
pgcc -ta=nvidia,time -acc -Minfo=accel sgemv2.c -o sgemv2 -O3
pgcc -ta=nvidia,time -acc -Minfo=accel sgemv_auto.c -o sgemvauto -O3
