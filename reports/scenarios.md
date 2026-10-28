| ID | Scenario | Expected | Observed | Result |
| --- | --- | --- | --- | --- |
| S1 | GET /api/health | 200, status ok, 3 modules, database ok | 200, status ok, modules ['diabetes', 'heart', 'breast'], database ok | Pass |
| S2 | GET /api/diseases | 200, modules in registry order diabetes, heart, breast | 200, diabetes, heart, breast | Pass |
| S3 | Heart module, typical positive case | 200, risk above the 0.5 line, screening flag on | 200, risk 0.868, flag True, 3 factors | Pass |
| S4 | Heart module, typical negative case | 200, risk below the screening line, flag off | 200, risk 0.083, flag False | Pass |
| S5 | Diabetes, glucose = 9999 | 422 with a field message naming the range | 422, Glucose: Must be between 40 and 300 mg/dL. | Pass |
| S6 | Diabetes, required Age omitted | 422, Age: Required | 422, Age: Required. | Pass |
| S7 | Diabetes, Age = 40.5 | 422, whole number required | 422, Age: Must be a whole number. | Pass |
| S8 | Diabetes, BMI = NaN in the JSON body | 422, value must be finite | 422, BMI: Must be a finite number. | Pass |
| S9 | Diabetes, unknown field Cholesterol | 422, unknown field rejected | 422, fields ['Cholesterol'] | Pass |
| S10 | Rejected requests S5 to S9 are not logged | history count unchanged | 2 before, 2 after | Pass |
| S11 | Diabetes, optional insulin left blank | 200 plus a 'not provided' warning | 200, risk 0.463, warnings 1: 2-hour serum insulin not provided: replaced by the training median/mode. | Pass |
| S12 | Diabetes, valid but unusual insulin = 880 | 200 plus an out-of-range-for-training warning | 200, 2-hour serum insulin = 880 lies outside the range typical of the training data (18.92 to 542.3); the estimate may be less reliable. | Pass |
| S13 | Unknown module asthma | 404 JSON error | 404, Unknown disease module 'asthma'. Available: ['breast', 'diabetes', 'heart'] | Pass |
| S14 | Malformed JSON body | 400 JSON error | 400, {'_payload': 'Request body must be valid JSON (Content-Type: application/json).'} | Pass |
| S15 | Body larger than 1 MB | 413 JSON error | 413 | Pass |
| S16 | GET on a POST-only route | 405 JSON error | 405, The method is not allowed for the requested URL. | Pass |
| S17 | Batch of 501 rows | 400, limit of 500 enforced | 400, {'rows': 'At most 500 rows per request.'} | Pass |
| S18 | Batch of 500 rows | 200, one result per row | 200, 500 results | Pass |
| S19 | Three valid breast-cancer predictions | history grows by exactly 3 | 4 before, 7 after | Pass |
| S20 | Hold-out metrics shown by the API (and UI) versus results.json | identical for every module and metric | 15 of 15 values identical | Pass |
| S21 | Served screening line and model name versus results.json | identical for every module | 3 of 3 modules identical | Pass |

21 of 21 scenarios passed.