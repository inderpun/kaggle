COMP = arc-prize-2026-arc-agi-2
DATA = data
KAGGLE = $(HOME)/.venvs/kaggle/bin/kaggle

.PHONY: data score submission clean

data:
	$(KAGGLE) competitions download -c $(COMP) -p $(DATA)
	cd $(DATA) && unzip -oq $(COMP).zip && rm -f $(COMP).zip
	@echo "Data ready in $(DATA)/"

score:
	cd src && python3 evaluate.py $(SOLVER)

clean:
	rm -rf $(DATA)/*.json $(DATA)/*.zip
