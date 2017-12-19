CC = gcc
OS := $(shell uname -s)
FLAGS = -Wall -O3 -std=gnu99 -g

ifeq ($(OS),Linux)
	LDFLAGS_MULTIPLE_EC = -Wl,-soname,C_call_boundary_dynamics.so
endif
ifeq ($(OS),Darwin)
	LDFLAGS_MULTIPLE_EC = -Wl,-install_name,C_call_boundary_dynamics.so
endif

all: C_call_boundary_dynamics.so

C_call_boundary_dynamics.so : C_call_boundary_dynamics.o
	$(CC) -shared $(LDFLAGS_MULTIPLE_EC) $(FLAGS) -o C_call_boundary_dynamics.so C_call_boundary_dynamics.o

C_call_boundary_dynamics.o : C_call_boundary_dynamics.c
	$(CC) -c -fPIC $(FLAGS) C_call_boundary_dynamics.c -o C_call_boundary_dynamics.o


clean:
	rm -f C_call_boundary_dynamics.so C_call_boundary_dynamics.o

