PYTHON ?= .venv/bin/python
RESULTS ?= ../kernelscope/results

.PHONY: demo test test-gpu package-demo doctor followup followup-plan

demo:
	bash scripts/demo.sh

test:
	$(PYTHON) -m pytest -q -p no:cacheprovider -m 'not gpu'

test-gpu:
	$(PYTHON) -m pytest -q -p no:cacheprovider -m gpu

package-demo:
	$(PYTHON) scripts/package_demo.py --results $(RESULTS)

doctor:
	$(PYTHON) -m kernelscope.cli serve doctor

FOLLOWUP_OUT ?= $(RESULTS)/serve_4090/followup_$(shell date -u +%Y%m%dT%H%M%SZ)

followup-plan:
	$(PYTHON) scripts/followup_campaign.py --out $(FOLLOWUP_OUT) --plan-only

followup:
	$(PYTHON) scripts/followup_campaign.py --out $(FOLLOWUP_OUT)
