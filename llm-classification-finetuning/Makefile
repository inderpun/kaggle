COMP = llm-classification-finetuning
DATA = data

.PHONY: data eda clean

data:
	kaggle competitions download -c $(COMP) -p $(DATA)
	cd $(DATA) && unzip -oq $(COMP).zip && rm -f $(COMP).zip
	@echo "Data ready in $(DATA)/"

eda:
	python3 src/eda.py

clean:
	rm -rf $(DATA)/*.csv $(DATA)/*.zip
