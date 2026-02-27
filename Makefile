BUILD_DIR := build
JOBS := $(shell nproc 2>/dev/null || sysctl -n hw.ncpu 2>/dev/null || echo 4)

.PHONY: all clean run app dmg

all:
	@mkdir -p $(BUILD_DIR)
	@cd $(BUILD_DIR) && cmake .. -DCMAKE_BUILD_TYPE=Release -G "Unix Makefiles" 2>&1 | tail -1
	@cmake --build $(BUILD_DIR) -j$(JOBS)

app:
	@mkdir -p $(BUILD_DIR)
	@cd $(BUILD_DIR) && cmake .. -DCMAKE_BUILD_TYPE=Release -DBUILD_APP_BUNDLE=ON -G "Unix Makefiles" 2>&1 | tail -1
	@cmake --build $(BUILD_DIR) -j$(JOBS)

dmg: app
	@bash scripts/create_dmg.sh

clean:
	@rm -rf $(BUILD_DIR)

run: all
	./$(BUILD_DIR)/microscope --model-dir models/ --model sdxs
