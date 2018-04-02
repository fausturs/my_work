#############################################################################
#							environment setting								#
#############################################################################
TARGET = a.out

BUILD_DIR = ./build
SRC_DIRS = ./src

SRCS := $(shell find $(SRC_DIRS) -name *.cpp -or -name *.c -or -name *.s)
OBJS := $(SRCS:%=$(BUILD_DIR)/%.o)
# -MMD -MP flags generate a .d file next to the .o file
DEPS := $(OBJS:.o=.d)

INC_DIRS := $(shell find $(SRC_DIRS) -type d)
INC_FLAGS := $(addprefix -I,$(INC_DIRS))

#############################################################################
#							other lib setting								#
#############################################################################
# mysql setting
MYSQL_INC_DIRS := /usr/include/mysql 
MYSQL_LIB_DIRS := /usr/lib/mysql 
INC_FLAGS += $(addprefix -I,$(MYSQL_INC_DIRS))
LIB_FLAGS += $(addprefix -L,$(MYSQL_LIB_DIRS))
	
#############################################################################
#							c++ compiler setting							#
#############################################################################
CXX = clang++
CPPFLAGS = -std=c++14 -O3 -MMD -MP $(INC_FLAGS)


#############################################################################
#							c compiler setting								#
#############################################################################
# CC = clang++
# CFLAGS = -MMD -MP $(INC_FLAGS)


#############################################################################
#						assembly compiler setting							#
#############################################################################
# AS = 
# ASFLAGS =

#############################################################################
#								link setting								#
#############################################################################
LD = clang++
LDFLAGS = $(LIB_FLAGS) -lmysqlclient -lz -lm


#############################################################################
#								make file									#
#############################################################################
# link
$(TARGET): $(OBJS)
	$(LD) $(OBJS) -o $@ $(LDFLAGS)

# assembly
$(BUILD_DIR)/%.s.o: %.s
	$(MKDIR_P) $(dir $@)
	$(AS) $(ASFLAGS) -c $< -o $@

# c source
$(BUILD_DIR)/%.c.o: %.c
	$(MKDIR_P) $(dir $@)
	$(CC) $(CFLAGS) -c $< -o $@

# c++ source
$(BUILD_DIR)/%.cpp.o: %.cpp
	$(MKDIR_P) $(dir $@)
	$(CXX) $(CPPFLAGS) -c $< -o $@


.PHONY: clean
clean:
	$(RM) -r $(TARGET) $(BUILD_DIR)

.PHONY: test
test:
	@echo $(INC_FLAGS)
	@echo $(LIB_FLAGS)
#############################################################################
#									others									#
#############################################################################
-include $(DEPS)

MKDIR_P ?= mkdir -p
