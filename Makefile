CONFIG_LOCAL  :=  ./config/Makefile.config

include $(CONFIG_LOCAL)

# 根据当前的环境修改gcc和cuda的版本
CXX                         :=  g++
CUDA_VER                    :=  11.3

# opencv和TensorRT的安装目录
OPENCV_INSTALL_DIR          :=  /usr/include/opencv4
TENSORRT_INSTALL_DIR        :=  /home/ztl/tensorrt_learning/TensorRT-8.4.1.5.Linux.x86_64-gnu.cuda-11.6.cudnn8.4/TensorRT-8.4.1.5

BUILD_PATH    :=  build
SRC_PATH      :=  src/cpp
INC_PATH      :=  include
CUDA_DIR      :=  /usr/local/cuda-$(CUDA_VER)

CXX_SRC       +=  $(wildcard $(SRC_PATH)/*.cpp)
KERNELS_SRC   :=  $(wildcard $(SRC_PATH)/*.cu)

APP_OBJS      :=  $(patsubst $(SRC_PATH)%, $(BUILD_PATH)%, $(CXX_SRC:.cpp=.cpp.o))
APP_OBJS      +=  $(patsubst $(SRC_PATH)%, $(BUILD_PATH)%, $(KERNELS_SRC:.cu=.cu.o))  

APP_MKS       :=  $(APP_OBJS:.o=.mk)

APP_DEPS      :=  $(CXX_SRC)
APP_DEPS      +=  $(KERNELS_SRC)
APP_DEPS      +=  $(wildcard $(SRC_PATH)/*.h)


CUCC          :=  $(CUDA_DIR)/bin/nvcc
CXXFLAGS      :=  -std=c++11 -pthread -fPIC
CUDAFLAGS     :=  --shared -Xcompiler -fPIC 

INCS          :=  -I $(CUDA_DIR)/include \
                  -I $(SRC_PATH) \
				  -I $(OPENCV_INSTALL_DIR) \
				  -I $(TENSORRT_INSTALL_DIR)/include \
				  -I $(INC_PATH)

LIBS          :=  -L "$(CUDA_DIR)/lib64" \
                  -L "$(TENSORRT_INSTALL_DIR)/lib" \
				  -lcudart -lcublas -lcudnn \
				  -lnvinfer -lnvonnxparser\
				  -lstdc++fs \
				  `pkg-config --libs opencv4`


ifeq ($(DEBUG),1)
CUDAFLAGS     +=  -g -O0
CXXFLAGS      +=  -g -O0
else
CUDAFLAGS     +=  -O3
CXXFLAGS      +=  -O3
endif

ifeq ($(SHOW_WARNING),1)
CUDAFLAGS     +=  -Wall -Wunused-function -Wunused-variable -Wfatal-errors
CXXFLAGS      +=  -Wall -Wunused-function -Wunused-variable -Wfatal-errors
else
CUDAFLAGS     +=  -w
CXXFLAGS      +=  -w
endif


ifeq (, $(shell which bear))
BEARCMD       :=
else
ifeq (bear 3.0.18, $(shell bear --version))
BEARCMD       := bear --output config/compile_commands.json --
else
BEARCMD       := bear -o config/compile_commands.json
endif
endif


all: 
	@mkdir -p bin
	@$(BEARCMD) $(MAKE) --no-print-directory $(APP)
	@echo finished building $@. Have fun!!

run:
	@$(MAKE) --no-print-directory update
	@./bin/$(APP)

update: $(APP)
	@echo finished updating $<

$(APP): $(APP_DEPS) $(APP_OBJS)
	@$(CXX) $(APP_OBJS) -o bin/$@ $(LIBS) $(INCS)

show: 
	@echo $(BUILD_PATH)
	@echo $(APP_DEPS)
	@echo $(INCS)
	@echo $(APP_OBJS)
	@echo $(APP_MKS)

clean:
	rm -rf $(APP)
	rm -rf build
	rm -rf config/compile_commands.json
	rm -rf bin

ifneq ($(MAKECMDGOALS), clean)
-include $(APP_MKS)
endif

# Compile CXX
$(BUILD_PATH)/%.cpp.o: $(SRC_PATH)/%.cpp 
	@echo Compile CXX $@
	@mkdir -p $(BUILD_PATH)
	@$(CXX) -o $@ -c $< $(CXXFLAGS) $(INCS)
$(BUILD_PATH)/%.cpp.mk: $(SRC_PATH)/%.cpp
	@echo Compile Dependence CXX $@
	@mkdir -p $(BUILD_PATH)
	@$(CXX) -M $< -MF $@ -MT $(@:.cpp.mk=.cpp.o) $(CXXFLAGS) $(INCS) 

# Compile CUDA
$(BUILD_PATH)/%.cu.o: $(SRC_PATH)/%.cu
	@echo Compile CUDA $@
	@mkdir -p $(BUILD_PATH)
	@$(CUCC) -o $@ -c $< $(CUDAFLAGS) $(INCS)
$(BUILD_PATH)/%.cu.mk: $(SRC_PATH)%.cu
	@echo Compile Dependence CUDA $@
	@mkdir -p $(BUILD_PATH)
	@$(CUCC) -M $< -MF $@ -MT $(@:.cu.mk=.cu.o) $(CUDAFLAGS)

.PHONY: all update show clean 
