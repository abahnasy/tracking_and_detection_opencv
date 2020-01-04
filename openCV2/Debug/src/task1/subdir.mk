################################################################################
# Automatically-generated file. Do not edit!
################################################################################

# Add inputs and outputs from these tool invocations to the build variables 
CPP_SRCS += \
../src/task1/HOGDescriptor.cpp \
../src/task1/task1.cpp 

OBJS += \
./src/task1/HOGDescriptor.o \
./src/task1/task1.o 

CPP_DEPS += \
./src/task1/HOGDescriptor.d \
./src/task1/task1.d 


# Each subdirectory must supply rules for building sources it contributes
src/task1/%.o: ../src/task1/%.cpp
	@echo 'Building file: $<'
	@echo 'Invoking: Cross G++ Compiler'
	g++ -I/usr/include/opencv -O0 -g3 -Wall -c -fmessage-length=0 -MMD -MP -MF"$(@:%.o=%.d)" -MT"$(@)" -o "$@" "$<"
	@echo 'Finished building: $<'
	@echo ' '


