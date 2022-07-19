**Prompt baselines**

(1) Inflection   

| lang      | plm                         |  acc.  |    edit distance |
|---------- |----------                   |------: | ----------------:|    
|eng        | t5 (prefix-tuning)          | 0.901  |       0.327      |
|eng        | gpt2-medium (prefix-tuning) | 0.915  |       0.291      |            
|deu        | mGPT (prefix-tuning)        | 0.716  |       1.203      |
|fra        | mGPT (prefix-tuning)        | 0.792  |       2.316      |



(2) Reinflection   

| lang      | plm                         |  acc.  |    edit distance |
|---------- |----------                   |------: | ----------------:|
|eng        | t5 (prefix-tuning)          | 0.903  |        0.367     |
|eng        | gpt2-medium (prefix-tuning) | 0.913  |        0.321     |
|fra        | mGPT (prefix-tuning)        | 0.787  |        0.753     |
|rus        | mGPT (prefix-tuning)        | 0.879  |        0.886     |
|tur        | mGPT (prefix-tuning)        | 0.811  |         ?        |   
|deu        | mGPT (prefix-tuning)        | 0.758  |        1.14      |
|heb        | mGPT (prefix-tuning)        | 0.160  |         ?        |
