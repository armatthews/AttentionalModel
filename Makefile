CC=g++
CNN_DIR = /home/austinma/git/ws15mt-cnn
#CNN_DIR = /home/austinma/git/cnn
EIGEN = /opt/tools/eigen-dev/
CNN_BUILD_DIR=$(CNN_DIR)/build
INCS=-I$(CNN_DIR) -I$(CNN_BUILD_DIR) -I$(EIGEN)
LIBS=-L$(CNN_BUILD_DIR)/cnn/
FINAL=-lcnn -lboost_regex -lboost_serialization
CFLAGS=-std=c++1y -Ofast -g -march=native -pipe
#CFLAGS=-std=c++1y -O0 -g -march=native -pipe
BINDIR=bin
SRCDIR=src

.PHONY: clean
all: $(BINDIR)/lstmlm $(BINDIR)/train $(BINDIR)/predict $(BINDIR)/sandbox $(BINDIR)/align

$(BINDIR)/sandbox: $(BINDIR)/sandbox.o
	mkdir -p $(BINDIR)
	$(CC) $(CFLAGS) $(LIBS) $(INCS) $(BINDIR)/sandbox.o -o $(BINDIR)/sandbox $(FINAL)

$(BINDIR)/train: $(BINDIR)/train.o $(BINDIR)/attentional.o $(BINDIR)/bitext.o
	mkdir -p $(BINDIR)
	$(CC) $(CFLAGS) $(LIBS) $(INCS) $(BINDIR)/train.o $(BINDIR)/attentional.o $(BINDIR)/bitext.o -o $(BINDIR)/train $(FINAL)

$(BINDIR)/predict: $(BINDIR)/predict.o $(BINDIR)/attentional.o $(BINDIR)/bitext.o
	mkdir -p $(BINDIR)
	$(CC) $(CFLAGS) $(LIBS) $(INCS) $(BINDIR)/predict.o $(BINDIR)/attentional.o $(BINDIR)/bitext.o -o $(BINDIR)/predict $(FINAL)

$(BINDIR)/align: $(BINDIR)/align.o $(BINDIR)/attentional.o $(BINDIR)/bitext.o
	mkdir -p $(BINDIR)
	$(CC) $(CFLAGS) $(LIBS) $(INCS) $(BINDIR)/align.o $(BINDIR)/attentional.o $(BINDIR)/bitext.o -o $(BINDIR)/align $(FINAL)

$(BINDIR)/sandbox.o: $(SRCDIR)/sandbox.cc src/utils.h src/kbestlist.h
	mkdir -p $(BINDIR)
	$(CC) $(CFLAGS) $(INCS) -c $(SRCDIR)/sandbox.cc -o $(BINDIR)/sandbox.o

$(BINDIR)/train.o: $(SRCDIR)/train.cc $(SRCDIR)/attentional.h $(SRCDIR)/bitext.h
	mkdir -p $(BINDIR)
	$(CC) $(CFLAGS) $(INCS) -c $(SRCDIR)/train.cc -o $(BINDIR)/train.o

$(BINDIR)/predict.o: $(SRCDIR)/predict.cc $(SRCDIR)/attentional.h
	mkdir -p $(BINDIR)
	$(CC) $(CFLAGS) $(INCS) -c $(SRCDIR)/predict.cc -o $(BINDIR)/predict.o

$(BINDIR)/align.o: $(SRCDIR)/align.cc $(SRCDIR)/attentional.h
	mkdir -p $(BINDIR)
	$(CC) $(CFLAGS) $(INCS) -c $(SRCDIR)/align.cc -o $(BINDIR)/align.o

$(BINDIR)/attentional.o: $(SRCDIR)/attentional.cc $(SRCDIR)/utils.h $(SRCDIR)/attentional.h $(SRCDIR)/bitext.h $(SRCDIR)/kbestlist.h
	mkdir -p $(BINDIR)
	$(CC) $(CFLAGS) $(INCS) -c $(SRCDIR)/attentional.cc -o $(BINDIR)/attentional.o

$(BINDIR)/bitext.o: $(SRCDIR)/bitext.cc $(SRCDIR)/bitext.h
	mkdir -p $(BINDIR)
	$(CC) $(CFLAGS) $(INCS) -c $(SRCDIR)/bitext.cc -o $(BINDIR)/bitext.o

$(BINDIR)/train.o:


$(BINDIR)/lstmlm: src/lstmlm.cc src/utils.h
	mkdir -p $(BINDIR)
	$(CC) $(CFLAGS) $(LIBS) $(INCS) $(SRCDIR)/lstmlm.cc -o $(BINDIR)/lstmlm $(FINAL)

clean:
	rm -rf $(BINDIR)/*
