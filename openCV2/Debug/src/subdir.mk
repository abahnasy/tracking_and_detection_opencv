################################################################################
# Automatically-generated file. Do not edit!
################################################################################

# Add inputs and outputs from these tool invocations to the build variables 
CPP_SRCS += \
../src/HogVisualization.cpp \
../src/LoadDataset_Task2.cpp \
../src/RandomForest.cpp \
../src/main.cpp \
../src/task1.cpp \
../src/task2.cpp 

OBJS += \
./src/HogVisualization.o \
./src/LoadDataset_Task2.o \
./src/RandomForest.o \
./src/main.o \
./src/task1.o \
./src/task2.o 

CPP_DEPS += \
./src/HogVisualization.d \
./src/LoadDataset_Task2.d \
./src/RandomForest.d \
./src/main.d \
./src/task1.d \
./src/task2.d 


# Each subdirectory must supply rules for building sources it contributes
src/%.o: ../src/%.cpp
	@echo 'Building file: $<'
	@echo 'Invoking: Cross G++ Compiler'
	g++ -I/usr/include/opencv -O0 -g3 -Wall -c -fmessage-length=0 -MMD -MP -MF"$(@:%.o=%.d)" -MT"$(@)" -o "$@" "$<"
	@echo 'Finished building: $<'
	@echo ' '


