# NE-AFE

Automated feature engineering for wind and photovoltaic power forecasting, combining domain-guided feature expansion with sequential temporal feature selection.

Python 3.11 or later.

```bash
git clone https://github.com/ChenZiXu-svg/NE-AFE.git
cd NE-AFE
pip install -r requirements.txt
python example.py
```

`example.py` demonstrates candidate construction, feature selection and Ridge forecasting on a synthetic time series. Replace the example data with your own chronologically ordered inputs and target.

Implementation: `neafe/pafa.py` (feature expansion), `neafe/gwbp.py` (feature selection).

MIT License.
