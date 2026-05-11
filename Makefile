CC=gcc
CFLAGS=-O2 -Wall -Wextra

all: run

run: challenge.c
	$(CC) $(CFLAGS) challenge.c -lm -o run

clean:
	rm -f run test_output.txt
