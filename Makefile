CC=g++
CNN_DIR = ./cnn
EIGEN = ./eigen
CNN_BUILD_DIR=$(CNN_DIR)/build
INCS=-I$(CNN_DIR) -I$(CNN_BUILD_DIR) -I$(EIGEN)
LIBS=-L$(CNN_BUILD_DIR)/dynet/ -L$(PREFIX)/lib
FINAL=-ldynet -lboost_regex -lboost_serialization -lboost_program_options -lrt -lpthread
#FINAL=-ldynet -ldynetcuda -lboost_regex -lboost_serialization -lboost_program_options -lcuda -lcudart -lcublas -lpthread -lrt
#CFLAGS=-std=c++11 -Ofast -g -march=native -pipe
CFLAGS=-std=c++11 -Wall -pedantic -O0 -g -pipe
BINDIR=bin
OBJDIR=obj
SRCDIR=src

.PHONY: clean
all: make_dirs $(BINDIR)/train $(BINDIR)/sample $(BINDIR)/align $(BINDIR)/loss $(BINDIR)/predict $(BINDIR)/sandbox

make_dirs:
	mkdir -p $(OBJDIR)
	mkdir -p $(BINDIR)

include $(wildcard $(OBJDIR)/*.d)

$(OBJDIR)/%.o: $(SRCDIR)/%.cc
	$(CC) $(CFLAGS) $(INCS) -c $< -o $@
	$(CC) -MM -MP -MT "$@" $(CFLAGS) $(INCS) $< > $(OBJDIR)/$*.d

$(BINDIR)/train: $(addprefix $(OBJDIR)/, train.o io.o translator.o tree_encoder.o encoder.o attention.o prior.o output.o rnng.o syntax_tree.o treelstm.o morphology.o mlp.o utils.o)
	$(CC) $(CFLAGS) $(LIBS) $(INCS) $^ -o $@ $(FINAL)

$(BINDIR)/sample: $(addprefix $(OBJDIR)/, sample.o io.o translator.o tree_encoder.o encoder.o attention.o prior.o output.o rnng.o utils.o syntax_tree.o morphology.o mlp.o treelstm.o)
	$(CC) $(CFLAGS) $(LIBS) $(INCS) $^ -o $@ $(FINAL)

$(BINDIR)/align: $(addprefix $(OBJDIR)/, align.o io.o translator.o tree_encoder.o encoder.o attention.o prior.o output.o rnng.o syntax_tree.o treelstm.o morphology.o mlp.o utils.o)
	$(CC) $(CFLAGS) $(LIBS) $(INCS) $^ -o $@ $(FINAL)

$(BINDIR)/loss: $(addprefix $(OBJDIR)/, loss.o io.o translator.o tree_encoder.o encoder.o attention.o prior.o output.o rnng.o syntax_tree.o treelstm.o morphology.o mlp.o utils.o)
	$(CC) $(CFLAGS) $(LIBS) $(INCS) $^ -o $@ $(FINAL)

$(BINDIR)/predict: $(addprefix $(OBJDIR)/, predict.o io.o translator.o tree_encoder.o encoder.o attention.o prior.o output.o rnng.o syntax_tree.o treelstm.o morphology.o kbestlist.o mlp.o utils.o)
	$(CC) $(CFLAGS) $(LIBS) $(INCS) $^ -o $@ $(FINAL)

$(BINDIR)/sandbox: $(addprefix $(OBJDIR)/, sandbox.o io.o translator.o tree_encoder.o encoder.o attention.o prior.o output.o rnng.o syntax_tree.o treelstm.o morphology.o kbestlist.o mlp.o utils.o)
	$(CC) $(CFLAGS) $(LIBS) $(INCS) $^ -o $@ $(FINAL)

clean:
	rm -rf $(BINDIR)/*
	rm -rf $(OBJDIR)/*
