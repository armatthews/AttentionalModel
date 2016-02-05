CC=g++
CNN_DIR = ./cnn
EIGEN = ./eigen
CNN_BUILD_DIR=$(CNN_DIR)/build
INCS=-I$(CNN_DIR) -I$(CNN_BUILD_DIR) -I$(EIGEN)
LIBS=-L$(CNN_BUILD_DIR)/cnn/
FINAL=-lcnn -lboost_regex -lboost_serialization -lboost_program_options -lrt -lpthread
#FINAL=-lcnn -lcnncuda -lboost_regex -lboost_serialization -lboost_program_options -lcuda -lcudart -lcublas
#CFLAGS=-std=c++11 -Ofast -g -march=native -pipe
CFLAGS=-std=c++11 -Wall -pedantic -O0 -g -pipe
BINDIR=bin
OBJDIR=obj
SRCDIR=src

.PHONY: clean
all: make_dirs $(BINDIR)/predict $(BINDIR)/sandbox $(BINDIR)/align $(BINDIR)/sample $(BINDIR)/loss $(BINDIR)/train $(BINDIR)/train-t2s $(BINDIR)/train-unify

make_dirs:
	mkdir -p $(OBJDIR)
	mkdir -p $(BINDIR)

include $(wildcard $(OBJDIR)/*.d)

$(OBJDIR)/%.o: $(SRCDIR)/%.cc
	$(CC) $(CFLAGS) $(INCS) -c $< -o $@
	$(CC) -MM -MP -MT "$@" $(CFLAGS) $(INCS) $< > $(OBJDIR)/$*.d

$(BINDIR)/sandbox: $(addprefix $(OBJDIR)/, sandbox.o attentional.o bitext.o decoder.o io.o utils.o syntax_tree.o treelstm.o)
	$(CC) $(CFLAGS) $(LIBS) $(INCS) $^ -o $@ $(FINAL)

$(BINDIR)/train-t2s: $(addprefix $(OBJDIR)/, train-t2s.o tree_encoder.o encoder.o attention.o output.o syntax_tree.o treelstm.o utils.o)
	$(CC) $(CFLAGS) $(LIBS) $(INCS) $^ -o $@ $(FINAL)

$(BINDIR)/train-unify: $(addprefix $(OBJDIR)/, train-unify.o tree_encoder.o encoder.o attention.o output.o syntax_tree.o treelstm.o utils.o)
	$(CC) $(CFLAGS) $(LIBS) $(INCS) $^ -o $@ $(FINAL)

$(BINDIR)/train: $(addprefix $(OBJDIR)/, train.o encoder.o attention.o output.o utils.o)
	$(CC) $(CFLAGS) $(LIBS) $(INCS) $^ -o $@ $(FINAL)

$(BINDIR)/predict: $(addprefix $(OBJDIR)/, predict.o decoder.o io.o bitext.o attentional.o syntax_tree.o kbestlist.o utils.o treelstm.o)
	$(CC) $(CFLAGS) $(LIBS) $(INCS) $^ -o $@ $(FINAL)

$(BINDIR)/align: $(addprefix $(OBJDIR)/, align.o decoder.o io.o bitext.o attentional.o syntax_tree.o utils.o treelstm.o)
	$(CC) $(CFLAGS) $(LIBS) $(INCS) $^ -o $@ $(FINAL)

$(BINDIR)/sample: $(addprefix $(OBJDIR)/, sample.o tree_encoder.o encoder.o attention.o output.o utils.o syntax_tree.o treelstm.o)
	$(CC) $(CFLAGS) $(LIBS) $(INCS) $^ -o $@ $(FINAL)

$(BINDIR)/loss: $(addprefix $(OBJDIR)/, loss.o attentional.o bitext.o decoder.o io.o utils.o syntax_tree.o treelstm.o)
	$(CC) $(CFLAGS) $(LIBS) $(INCS) $^ -o $@ $(FINAL)

clean:
	rm -rf $(BINDIR)/*
	rm -rf $(OBJDIR)/*
