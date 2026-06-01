import pandas as pd

class DataframeProperties:
    def __init__(self, path: str):
        self.dt = pd.read_csv(path)
        self.dt["date"] = pd.to_datetime(self.dt["date"], format = "%d-%m-%Y")

    def max_min_date(self) -> tuple:
        valid_dates = self.dt["date"].dropna()
        
        if valid_dates.empty:
            return None, None 
        
        max_date = valid_dates.max().strftime("%d-%m-%Y")  
        min_date = valid_dates.min().strftime("%d-%m-%Y")  
        return max_date, min_date
        
    
    def date_list(self) -> list:
        if self.dt["date"].isnull().all():
            return []

        return self.dt["date"].dt.strftime('%d-%m-%Y').value_counts().keys().to_list()

    def lin_dt(self) -> int:
        return len(self.dt)

class PromptYN:
    def __init__(self, msg: str):
        self.msg = msg
        self.res = self.get_response()

    def get_response(self) -> str:
        while True:
            response = input(f"\n{self.msg} (Y/n): ").strip().upper()
            if response in ['Y', 'N', '']:
                return response
            else:
                print(f"O valor {response} não é aceito, apenas 'Y' ou 'n'.")

    def response(self) -> bool:
        return self.res == 'Y'