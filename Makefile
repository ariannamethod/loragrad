# loragrad — phase 1 (mechanism, not training)
#
# Two artifacts:
#   loragrad_standalone.o — does NOT pull in notorch (lg_signature_from_grads
#                           is left undefined; smoke test uses text sketcher)
#   loragrad.o            — full version, links against /opt/homebrew/libnotorch.a
#
# smoke_test uses loragrad_standalone.o so it builds and runs without
# requiring a notorch model. Phase 2 will add a notorch-backed example.

CC      ?= cc
CFLAGS  ?= -O2 -std=c11 -Wall -Wextra -Wno-unused-parameter
CFLAGS  += -I.
NT_INC   = -I.
NT_LIB   = /opt/homebrew/lib/libnotorch.a
NT_FRAME = -framework Accelerate
LDFLAGS ?=
LDLIBS  ?= -lm

.PHONY: all clean smoke train

all: loragrad.o loragrad_standalone.o examples/smoke_test examples/train_loragrad

# Standalone object — built without notorch headers. Provides text/buffer
# sketchers and the full field/voting machinery; lg_signature_from_grads is
# stubbed to a safe no-op so the symbol exists.
loragrad_standalone.o: loragrad.c loragrad.h
	$(CC) $(CFLAGS) -DLG_STANDALONE -c -o $@ loragrad.c

# Notorch-linked object — the full version. Used when the caller has a
# model on the active tape and wants gradient-based signatures.
loragrad.o: loragrad.c loragrad.h
	$(CC) $(CFLAGS) $(NT_INC) -c -o $@ loragrad.c

examples/smoke_test: examples/smoke_test.c loragrad_standalone.o loragrad.h
	$(CC) $(CFLAGS) -o $@ examples/smoke_test.c loragrad_standalone.o $(LDLIBS)

examples/train_loragrad: examples/train_loragrad.c loragrad.o loragrad.h
	$(CC) $(CFLAGS) $(NT_INC) -o $@ examples/train_loragrad.c loragrad.o $(NT_LIB) $(NT_FRAME) $(LDLIBS)

smoke: examples/smoke_test
	./examples/smoke_test

train: examples/train_loragrad
	cd $(CURDIR) && ./examples/train_loragrad --routed

clean:
	rm -f loragrad.o loragrad_standalone.o examples/smoke_test examples/train_loragrad
