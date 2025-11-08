# Barretina Negra
Run streamlit (inside virtual environment) with:
```bash
streamlit run dashboard/conflictivity_dashboard.py --server.headless true --server.port 85029
```


## Field in Client Logs that relate to APs:
`associated_device` — matches the AP's macaddr field
Example: Client has `"associated_device": "AP_8e2d9933ec92"` → matches AP's `"macaddr": "AP_8e2d9933ec92"`

