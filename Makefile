CC=g++
DYNET_DIR = ./dynet
EIGEN = ./eigen
DYNET_BUILD_DIR=$(DYNET_DIR)/build
INCS=-I$(DYNET_DIR) -I$(DYNET_BUILD_DIR) -I$(EIGEN)
LIBS=-L$(DYNET_BUILD_DIR)/dynet/ -L$(PREFIX)/lib
FINAL=-ldynet -lboost_regex -lboost_serialization -lboost_program_options -lrt -lpthread
#FINAL=-lgdynet -lboost_regex -lboost_serialization -lboost_program_options -lcudart -lcublas -lpthread -lrt
CFLAGS=-std=c++11 -Ofast -g -pipe
#CFLAGS=-std=c++11 -Wall -pedantic -O0 -g -pipe
BINDIR=bin
OBJDIR=obj
SRCDIR=src

.PHONY: clean
all: make_dirs $(BINDIR)/train $(BINDIR)/sample $(BINDIR)/align $(BINDIR)/loss $(BINDIR)/predict $(BINDIR)/residual

make_dirs:
	mkdir -p $(OBJDIR)
	mkdir -p $(BINDIR)

include $(wildcard $(OBJDIR)/*.d)

$(OBJDIR)/%.o: $(SRCDIR)/%.cc
	$(CC) $(CFLAGS) $(INCS) -c $< -o $@
	$(CC) -MM -MP -MT "$@" $(CFLAGS) $(INCS) $< > $(OBJDIR)/$*.d

$(BINDIR)/train: $(addprefix $(OBJDIR)/, train.o io.o translator.o tree_encoder.o encoder.o attention.o prior.o output.o rnng.o syntax_tree.o embedder.o mlp.o utils.o)
	$(CC) $(CFLAGS) $(LIBS) $(INCS) $^ -o $@ $(FINAL)

$(BINDIR)/residual: $(addprefix $(OBJDIR)/, residual.o io.o translator.o tree_encoder.o encoder.o attention.o prior.o output.o rnng.o syntax_tree.o embedder.o mlp.o utils.o)
	$(CC) $(CFLAGS) $(LIBS) $(INCS) $^ -o $@ $(FINAL)

$(BINDIR)/sample: $(addprefix $(OBJDIR)/, sample.o io.o translator.o tree_encoder.o encoder.o attention.o prior.o output.o rnng.o utils.o syntax_tree.o embedder.o mlp.o)
	$(CC) $(CFLAGS) $(LIBS) $(INCS) $^ -o $@ $(FINAL)

$(BINDIR)/align: $(addprefix $(OBJDIR)/, align.o io.o translator.o tree_encoder.o encoder.o attention.o prior.o output.o rnng.o syntax_tree.o embedder.o mlp.o utils.o)
	$(CC) $(CFLAGS) $(LIBS) $(INCS) $^ -o $@ $(FINAL)

$(BINDIR)/loss: $(addprefix $(OBJDIR)/, loss.o io.o translator.o tree_encoder.o encoder.o attention.o prior.o output.o rnng.o syntax_tree.o embedder.o mlp.o utils.o)
	$(CC) $(CFLAGS) $(LIBS) $(INCS) $^ -o $@ $(FINAL)

$(BINDIR)/predict: $(addprefix $(OBJDIR)/, predict.o io.o translator.o tree_encoder.o encoder.o attention.o prior.o output.o rnng.o syntax_tree.o embedder.o kbestlist.o mlp.o utils.o)
	$(CC) $(CFLAGS) $(LIBS) $(INCS) $^ -o $@ $(FINAL)

clean:
	rm -rf $(BINDIR)/*
	rm -rf $(OBJDIR)/*
