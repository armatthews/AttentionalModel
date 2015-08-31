CC=g++
CNN_DIR = ./cnn
EIGEN = ./eigen
CNN_BUILD_DIR=$(CNN_DIR)/build
INCS=-I$(CNN_DIR) -I$(CNN_BUILD_DIR) -I$(EIGEN)
LIBS=-L$(CNN_BUILD_DIR)/cnn/
FINAL=-lcnn -lboost_regex -lboost_serialization -lboost_program_options
#FINAL=-lcnn -lcnncuda -lboost_regex -lboost_serialization -lboost_program_options -lcuda -lcudart -lcublas
#CFLAGS=-std=c++11 -Ofast -g -march=native -pipe
CFLAGS=-std=c++11 -Wall -pedantic -O0 -g -pipe
BINDIR=bin
OBJDIR=obj
SRCDIR=src

.PHONY: clean
all: make_dirs $(BINDIR)/train $(BINDIR)/predict $(BINDIR)/sandbox $(BINDIR)/align $(BINDIR)/sample $(BINDIR)/loss

make_dirs:
	mkdir -p $(OBJDIR)
	mkdir -p $(BINDIR)

include $(wildcard $(OBJDIR)/*.d)

$(OBJDIR)/%.o: $(SRCDIR)/%.cc
	$(CC) $(CFLAGS) $(INCS) -c $< -o $@
	$(CC) -MM -MP -MT "$@" $(CFLAGS) $(INCS) $< > $(OBJDIR)/$*.d

$(BINDIR)/sandbox: $(addprefix $(OBJDIR)/, sandbox.o syntax_tree.o utils.o)
	$(CC) $(CFLAGS) $(LIBS) $(INCS) $^ -o $@ $(FINAL)

$(BINDIR)/train: $(addprefix $(OBJDIR)/, train.o decoder.o attentional.o bitext.o utils.o syntax_tree.o treelstm.o)
	$(CC) $(CFLAGS) $(LIBS) $(INCS) $^ -o $@ $(FINAL)

$(BINDIR)/predict: $(addprefix $(OBJDIR)/, predict.o decoder.o bitext.o attentional.o syntax_tree.o kbestlist.o utils.o treelstm.o)
	$(CC) $(CFLAGS) $(LIBS) $(INCS) $^ -o $@ $(FINAL)

$(BINDIR)/align: $(addprefix $(OBJDIR)/, align.o decoder.o bitext.o attentional.o syntax_tree.o utils.o treelstm.o)
	$(CC) $(CFLAGS) $(LIBS) $(INCS) $^ -o $@ $(FINAL)

$(BINDIR)/sample: $(addprefix $(OBJDIR)/, sample.o attentional.o bitext.o decoder.o utils.o syntax_tree.o treelstm.o)
	$(CC) $(CFLAGS) $(LIBS) $(INCS) $^ -o $@ $(FINAL)

$(BINDIR)/loss: $(addprefix $(OBJDIR)/, loss.o attentional.o bitext.o decoder.o utils.o syntax_tree.o treelstm.o)
	$(CC) $(CFLAGS) $(LIBS) $(INCS) $^ -o $@ $(FINAL)

clean:
	rm -rf $(BINDIR)/*
	rm -rf $(OBJDIR)/*
