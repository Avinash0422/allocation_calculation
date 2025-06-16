import streamlit as st
import pandas as pd

def generate_tag_logic(max_tag=34):
    """Generate tag_logic dictionary dynamically up to max_tag, extrapolating beyond tag 25."""
    tag_logic = {
        1:  [1, 0, 0, 0, 0],
        2:  [1, 1, 0, 0, 0],
        3:  [1, 1, 1, 0, 0],
        4:  [1, 1, 1, 1, 0],
        5:  [1, 1, 1, 1, 1],
        6:  [2, 1, 1, 1, 1],
        7:  [2, 2, 1, 1, 1],
        8:  [2, 2, 2, 1, 1],
        9:  [2, 2, 2, 2, 1],
        10: [2, 2, 2, 2, 2],
        11: [3, 2, 2, 2, 2],
        12: [3, 3, 2, 2, 2],
        13: [3, 3, 3, 2, 2],
        14: [3, 3, 3, 3, 2],
        15: [3, 3, 3, 3, 3],
        16: [4, 3, 3, 3, 3],
        17: [4, 4, 3, 3, 3],
        18: [4, 4, 4, 3, 3],
        19: [4, 4, 4, 4, 3],
        20: [4, 4, 4, 4, 4],
        21: [5, 4, 4, 4, 4],
        22: [5, 5, 4, 4, 4],
        23: [5, 5, 5, 4, 4],
        24: [5, 5, 5, 5, 4],
        25: [5, 5, 5, 5, 5]
    }
    if max_tag >= 26:
        tag_logic.update({tag: [5, 5, 5, 5, 5] for tag in range(26, max_tag + 1)})
    return tag_logic

# Define allocation tables for G, D, and B strategies
allocation_tables = {
    'G': [
        {'Min': 1500000, 'Max': 4500000, 'Tags': 1, 'Capital': 2000000, 'SL%': 1},
        {'Min': 4500000, 'Max': 7500000, 'Tags': 2, 'Capital': 4000000, 'SL%': 1},
        {'Min': 7500000, 'Max': 10500000, 'Tags': 3, 'Capital': 6000000, 'SL%': 1},
        {'Min': 10500000, 'Max': 14500000, 'Tags': 4, 'Capital': 8000000, 'SL%': 1},
        {'Min': 14500000, 'Max': 18500000, 'Tags': 5, 'Capital': 10000000, 'SL%': 1},
        {'Min': 18500000, 'Max': 22500000, 'Tags': 6, 'Capital': 12000000, 'SL%': 1},
        {'Min': 22500000, 'Max': 26500000, 'Tags': 7, 'Capital': 14000000, 'SL%': 1},
        {'Min': 26500000, 'Max': 30500000, 'Tags': 8, 'Capital': 16000000, 'SL%': 1},
        {'Min': 30500000, 'Max': 34500000, 'Tags': 9, 'Capital': 18000000, 'SL%': 1},
        {'Min': 34500000, 'Max': 38500000, 'Tags': 10, 'Capital': 20000000, 'SL%': 1},
        {'Min': 38500000, 'Max': 42500000, 'Tags': 11, 'Capital': 22000000, 'SL%': 1},
        {'Min': 42500000, 'Max': 46500000, 'Tags': 12, 'Capital': 24000000, 'SL%': 1},
        {'Min': 46500000, 'Max': 50500000, 'Tags': 13, 'Capital': 26000000, 'SL%': 1},
        {'Min': 50500000, 'Max': 54500000, 'Tags': 14, 'Capital': 28000000, 'SL%': 1},
        {'Min': 54500000, 'Max': 58500000, 'Tags': 15, 'Capital': 30000000, 'SL%': 1},
        {'Min': 58500000, 'Max': 62500000, 'Tags': 16, 'Capital': 32000000, 'SL%': 1},
        {'Min': 62500000, 'Max': 66500000, 'Tags': 17, 'Capital': 34000000, 'SL%': 1},
        {'Min': 66500000, 'Max': 70500000, 'Tags': 18, 'Capital': 36000000, 'SL%': 1},
        {'Min': 70500000, 'Max': 74500000, 'Tags': 19, 'Capital': 38000000, 'SL%': 1},
        {'Min': 74500000, 'Max': 78500000, 'Tags': 20, 'Capital': 40000000, 'SL%': 1},
        {'Min': 78500000, 'Max': 82500000, 'Tags': 21, 'Capital': 42000000, 'SL%': 1},
        {'Min': 82500000, 'Max': 86500000, 'Tags': 22, 'Capital': 44000000, 'SL%': 1},
        {'Min': 86500000, 'Max': 90500000, 'Tags': 23, 'Capital': 46000000, 'SL%': 1},
        {'Min': 90500000, 'Max': 94500000, 'Tags': 24, 'Capital': 48000000, 'SL%': 1},
        {'Min': 94500000, 'Max': 98500000, 'Tags': 25, 'Capital': 50000000, 'SL%': 1},
        {'Min': 98500000, 'Max': 102500000, 'Tags': 26, 'Capital': 52000000, 'SL%': 1}
    ],
    'D': [
        {'Min': 4500000, 'Max': 9000000, 'Tags': 1, 'Capital': 4000000, 'SL%': 0.005},
        {'Min': 9000000, 'Max': 15000000, 'Tags': 2, 'Capital': 8000000, 'SL%': 0.005},
        {'Min': 15000000, 'Max': 21000000, 'Tags': 3, 'Capital': 12000000, 'SL%': 0.005},
        {'Min': 21000000, 'Max': 27000000, 'Tags': 4, 'Capital': 16000000, 'SL%': 0.005},
        {'Min': 27000000, 'Max': 33000000, 'Tags': 5, 'Capital': 20000000, 'SL%': 0.005},
        {'Min': 33000000, 'Max': 39000000, 'Tags': 6, 'Capital': 24000000, 'SL%': 0.005},
        {'Min': 39000000, 'Max': 45000000, 'Tags': 7, 'Capital': 28000000, 'SL%': 0.005},
        {'Min': 45000000, 'Max': 51000000, 'Tags': 8, 'Capital': 32000000, 'SL%': 0.005},
        {'Min': 51000000, 'Max': 57000000, 'Tags': 9, 'Capital': 36000000, 'SL%': 0.005},
        {'Min': 57000000, 'Max': 63000000, 'Tags': 10, 'Capital': 40000000, 'SL%': 0.005},
        {'Min': 63000000, 'Max': 69000000, 'Tags': 11, 'Capital': 44000000, 'SL%': 0.005},
        {'Min': 69000000, 'Max': 75000000, 'Tags': 12, 'Capital': 48000000, 'SL%': 0.005},
        {'Min': 75000000, 'Max': 81000000, 'Tags': 13, 'Capital': 52000000, 'SL%': 0.005},
        {'Min': 81000000, 'Max': 87000000, 'Tags': 14, 'Capital': 56000000, 'SL%': 0.005},
        {'Min': 87000000, 'Max': 93000000, 'Tags': 15, 'Capital': 60000000, 'SL%': 0.005},
        {'Min': 93000000, 'Max': 99000000, 'Tags': 16, 'Capital': 64000000, 'SL%': 0.005},
        {'Min': 99000000, 'Max': 105000000, 'Tags': 17, 'Capital': 68000000, 'SL%': 0.005}
    ],
    'B': [
        {'Min': 0, 'Max': 3000000, 'Tags': 1, 'Capital': 2000000, 'SL%': 0.008},
        {'Min': 3000000, 'Max': 6000000, 'Tags': 2, 'Capital': 4000000, 'SL%': 0.008},
        {'Min': 6000000, 'Max': 9000000, 'Tags': 3, 'Capital': 6000000, 'SL%': 0.008},
        {'Min': 9000000, 'Max': 12000000, 'Tags': 4, 'Capital': 8000000, 'SL%': 0.008},
        {'Min': 12000000, 'Max': 15000000, 'Tags': 5, 'Capital': 10000000, 'SL%': 0.008},
        {'Min': 15000000, 'Max': 18000000, 'Tags': 6, 'Capital': 12000000, 'SL%': 0.008},
        {'Min': 18000000, 'Max': 21000000, 'Tags': 7, 'Capital': 14000000, 'SL%': 0.008},
        {'Min': 21000000, 'Max': 24000000, 'Tags': 8, 'Capital': 16000000, 'SL%': 0.008},
        {'Min': 24000000, 'Max': 27000000, 'Tags': 9, 'Capital': 18000000, 'SL%': 0.008},
        {'Min': 27000000, 'Max': 30000000, 'Tags': 10, 'Capital': 20000000, 'SL%': 0.008},
        {'Min': 30000000, 'Max': 33000000, 'Tags': 11, 'Capital': 22000000, 'SL%': 0.008},
        {'Min': 33000000, 'Max': 36000000, 'Tags': 12, 'Capital': 24000000, 'SL%': 0.008},
        {'Min': 36000000, 'Max': 39000000, 'Tags': 13, 'Capital': 26000000, 'SL%': 0.008},
        {'Min': 39000000, 'Max': 42000000, 'Tags': 14, 'Capital': 28000000, 'SL%': 0.008},
        {'Min': 42000000, 'Max': 45000000, 'Tags': 15, 'Capital': 30000000, 'SL%': 0.008},
        {'Min': 45000000, 'Max': 48000000, 'Tags': 16, 'Capital': 32000000, 'SL%': 0.008},
        {'Min': 48000000, 'Max': 51000000, 'Tags': 17, 'Capital': 34000000, 'SL%': 0.008},
        {'Min': 51000000, 'Max': 54000000, 'Tags': 18, 'Capital': 36000000, 'SL%': 0.008},
        {'Min': 54000000, 'Max': 57000000, 'Tags': 19, 'Capital': 38000000, 'SL%': 0.008},
        {'Min': 57000000, 'Max': 60000000, 'Tags': 20, 'Capital': 40000000, 'SL%': 0.008},
        {'Min': 60000000, 'Max': 63000000, 'Tags': 21, 'Capital': 42000000, 'SL%': 0.008},
        {'Min': 63000000, 'Max': 66000000, 'Tags': 22, 'Capital': 44000000, 'SL%': 0.008},
        {'Min': 66000000, 'Max': 69000000, 'Tags': 23, 'Capital': 46000000, 'SL%': 0.008},
        {'Min': 69000000, 'Max': 72000000, 'Tags': 24, 'Capital': 48000000, 'SL%': 0.008},
        {'Min': 72000000, 'Max': 75000000, 'Tags': 25, 'Capital': 50000000, 'SL%': 0.008},
        {'Min': 75000000, 'Max': 78000000, 'Tags': 26, 'Capital': 52000000, 'SL%': 0.008},
        {'Min': 78000000, 'Max': 81000000, 'Tags': 27, 'Capital': 54000000, 'SL%': 0.008},
        {'Min': 81000000, 'Max': 84000000, 'Tags': 28, 'Capital': 56000000, 'SL%': 0.008},
        {'Min': 84000000, 'Max': 87000000, 'Tags': 29, 'Capital': 58000000, 'SL%': 0.008},
        {'Min': 87000000, 'Max': 90000000, 'Tags': 30, 'Capital': 60000000, 'SL%': 0.008},
        {'Min': 90000000, 'Max': 93000000, 'Tags': 31, 'Capital': 62000000, 'SL%': 0.008},
        {'Min': 93000000, 'Max': 96000000, 'Tags': 32, 'Capital': 64000000, 'SL%': 0.008},
        {'Min': 96000000, 'Max': 99000000, 'Tags': 33, 'Capital': 66000000, 'SL%': 0.008},
        {'Min': 99000000, 'Max': 102000000, 'Tags': 34, 'Capital': 68000000, 'SL%': 0.008}
    ]
}

def process_csv(file):
    """Process the strategy CSV file to count the number of times each user appears in each StrategyTag."""
    try:
        df = pd.read_csv(file, skiprows=5)
        df.columns = df.columns.str.strip()
        required_cols = ["StrategyTag", "User Account"]
        for col in required_cols:
            if col not in df.columns:
                st.error(f"Error: '{col}' column not found in dataset.")
                return None
        df = df[required_cols]
        df = df[~df["StrategyTag"].isin(["DEFAULT", "ZZZ"])]
        valid_strategies = ['B1', 'B2', 'B3', 'B4', 'B5', 'D1', 'D2', 'D3', 'D4', 'D5', 'G1', 'G2', 'G3', 'G4', 'G5']
        df = df[df["StrategyTag"].isin(valid_strategies)]
        user_accounts_split = df["User Account"].str.split(";", expand=True)
        df_melted = user_accounts_split.melt(ignore_index=False, var_name="User Column", value_name="User Account")
        df_melted.dropna(subset=["User Account"], inplace=True)
        df_melted.reset_index(inplace=True)
        df_melted = df_melted.merge(df[["StrategyTag"]], left_on="index", right_index=True, how="left")
        df_melted[["User Type", "User Value"]] = df_melted["User Account"].str.split("=", expand=True)
        df_melted["User Type"] = df_melted["User Type"].astype(str)
        df_melted["User Value"] = pd.to_numeric(df_melted["User Value"], errors="coerce").fillna(1).astype(int)
        df_counts = df_melted.groupby(["StrategyTag", "User Type"])["User Value"].sum().reset_index()
        df_pivot = df_counts.pivot(index="StrategyTag", columns="User Type", values="User Value").fillna(0).astype(int)
        df_pivot = df_pivot.reset_index()
        return df_pivot
    except pd.errors.EmptyDataError:
        st.error("Error: The file is empty.")
        return None
    except pd.errors.ParserError:
        st.error("Error: Unable to parse the file. Ensure it's a valid CSV.")
        return None
    except Exception as e:
        st.error(f"An unexpected error occurred: {e}")
        return None

def get_allocation_tag(allocation, series):
    """Map an allocation to a Tag, Capital, and Expected Max Loss based on the allocation table and series."""
    if series not in ['B', 'D', 'G']:
        return None, None, None, f"Invalid series: {series}"
    
    capital_deployed_tags = {
        'G': [
            (2000000, 1), (4000000, 2), (6000000, 3), (8000000, 4), (10000000, 5),
            (12000000, 6), (14000000, 7), (16000000, 8), (18000000, 9), (20000000, 10),
            (22000000, 11), (24000000, 12), (26000000, 13), (28000000, 14), (30000000, 15),
            (32000000, 16), (34000000, 17), (36000000, 18), (38000000, 19), (40000000, 20),
            (42000000, 21), (44000000, 22), (46000000, 23), (48000000, 24), (50000000, 25),
            (52000000, 26)
        ],
        'D': [
            (4000000, 1), (8000000, 2), (12000000, 3), (16000000, 4), (20000000, 5),
            (24000000, 6), (28000000, 7), (32000000, 8), (36000000, 9), (40000000, 10),
            (44000000, 11), (48000000, 12), (52000000, 13), (56000000, 14), (60000000, 15),
            (64000000, 16), (68000000, 17)
        ],
        'B': [
            (2000000, 1), (4000000, 2), (6000000, 3), (8000000, 4), (10000000, 5),
            (12000000, 6), (14000000, 7), (16000000, 8), (18000000, 9), (20000000, 10),
            (22000000, 11), (24000000, 12), (26000000, 13), (28000000, 14), (30000000, 15),
            (32000000, 16), (34000000, 17), (36000000, 18), (38000000, 19), (40000000, 20),
            (42000000, 21), (44000000, 22), (46000000, 23), (48000000, 24), (50000000, 25),
            (52000000, 26), (54000000, 27), (56000000, 28), (58000000, 29), (60000000, 30),
            (62000000, 31), (64000000, 32), (66000000, 33), (68000000, 34)
        ]
    }
    
    for capital, tag in capital_deployed_tags[series]:
        if allocation == capital:
            expected_max_loss = int(capital * (1 if series == 'G' else 0.005 if series == 'D' else 0.008))
            return int(tag), int(capital), expected_max_loss, None

    table = allocation_tables[series]
    last_row = table[-1]
    max_allocation = last_row['Max']
    last_tag = last_row['Tags']
    last_capital = last_row['Capital']
    sl_percent = last_row['SL%']
    capital_increment = 4000000 if series in ['D', 'G'] else 3000000
    range_width = max_allocation - table[-2]['Max'] if len(table) > 1 else 4000000 if series in ['D', 'G'] else 3000000

    for row in table:
        if row['Min'] <= allocation <= row['Max']:
            expected_max_loss = int(row['Capital'] * row['SL%'])
            return int(row['Tags']), int(row['Capital']), expected_max_loss, None

    if allocation > max_allocation:
        increments = (allocation - max_allocation) // range_width
        new_tag = int(last_tag + increments)
        new_capital = int(last_capital + increments * capital_increment)
        expected_max_loss = int(new_capital * sl_percent)
        return new_tag, new_capital, expected_max_loss, f"Extrapolated tag {new_tag} for allocation {allocation}"

    return None, None, None, f"No range match found for allocation {allocation} in series {series}."

def allocate_strategies(file, user_types, strategies, series):
    """Process the user settings CSV file to compute allocations, tags, and validate max loss."""
    try:
        df = pd.read_csv(file, skiprows=6)
    except pd.errors.EmptyDataError:
        st.error("Error: The user settings file is empty.")
        return None, None
    except pd.errors.ParserError:
        st.error("Error: Unable to parse the user settings file. Ensure it's a valid CSV.")
        return None, None
    except Exception as e:
        st.error(f"An unexpected error occurred while reading user settings file: {e}")
        return None, None

    df.columns = df.columns.str.strip()
    required_cols = ['User ID', 'Telegram ID(s)', 'Max Loss']
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        st.error(f"Error: The following required columns are missing in the user settings file: {missing_cols}")
        return None, None

    df['User ID'] = df['User ID'].astype(str)
    user_types = [str(user_type) for user_type in user_types]
    df['Telegram ID(s)'] = pd.to_numeric(df['Telegram ID(s)'], errors='coerce').fillna(0).astype(int)
    df['ALLOCATION'] = df['Telegram ID(s)'] * 100
    df['Max Loss'] = pd.to_numeric(df['Max Loss'], errors='coerce').fillna(0).astype(int)

    max_allocation = df['ALLOCATION'].max()
    table = allocation_tables[series]
    last_row = table[-1]
    max_table_allocation = last_row['Max']
    range_width = max_table_allocation - table[-2]['Max'] if len(table) > 1 else 4000000 if series in ['D', 'G'] else 3000000
    max_tag = last_row['Tags']
    if max_allocation > max_table_allocation:
        increments = (max_allocation - max_table_allocation) // range_width
        max_tag += increments

    tag_logic = generate_tag_logic(max_tag=max_tag)
    tag_allocation_rules = {'G': {}, 'D': {}, 'B': {}}
    for tag, values in tag_logic.items():
        tag_allocation_rules['G'][tag] = {str(i+1): value for i, value in enumerate(values)}
        tag_allocation_rules['D'][tag] = {str(i+1): value for i, value in enumerate(values)}
        tag_allocation_rules['B'][tag] = {str(i+1): value for i, value in enumerate(values)}

    allocation_dict = {}
    user_info = {}
    for _, row in df.iterrows():
        user_id = str(row['User ID'])
        allocation = row['ALLOCATION']
        max_loss = row['Max Loss']
        tag, capital, expected_max_loss, error = get_allocation_tag(allocation, series)
        allocation_dict[user_id] = allocation
        user_info[user_id] = {
            'allocation': allocation,
            'tag': tag,
            'max_loss': max_loss,
            'expected_max_loss': expected_max_loss,
            'capital': capital,
            'error': error
        }
        if error:
            st.warning(f"User {user_id}: {error}")

    df_filtered = df[['User ID', 'ALLOCATION', 'Max Loss']].copy()
    df_filtered.rename(columns={'User ID': 'userId'}, inplace=True)
    df_filtered['userId'] = df_filtered['userId'].astype(str)
    all_users_df = pd.DataFrame({'userId': user_types})
    df_filtered = all_users_df.merge(df_filtered, on="userId", how="left").fillna({'ALLOCATION': 0, 'Max Loss': 0})

    allocation_df = pd.DataFrame({'userId': user_types})
    allocation_df['ALLOCATION'] = df_filtered['ALLOCATION']
    allocation_df['Max Loss'] = df_filtered['Max Loss']
    for strategy in strategies:
        allocation_df[strategy] = 0

    for user_type in user_types:
        if user_type in user_info and user_info[user_type]['tag'] in tag_allocation_rules[series]:
            rule = tag_allocation_rules[series][user_info[user_type]['tag']]
            for idx, strategy in enumerate(strategies, 1):
                allocation_df.loc[allocation_df['userId'] == user_type, strategy] = rule[str(idx)]

    return allocation_df, (allocation_dict, user_info, tag_allocation_rules)

def main():
    st.title("Strategy Allocation and Max Loss Validation")
    
    stoxo_file = st.file_uploader("Upload Strategy CSV file", type=["csv"])
    allocation_file = st.file_uploader("Upload User Settings CSV file", type=["csv"])

    if stoxo_file is not None and allocation_file is not None:
        with st.spinner("Processing files..."):
            stoxo_df = process_csv(stoxo_file)
            if stoxo_df is None:
                return

            valid_strategies = ['B1', 'B2', 'B3', 'B4', 'B5', 'D1', 'D2', 'D3', 'D4', 'D5', 'G1', 'G2', 'G3', 'G4', 'G5']
            available_strategies = stoxo_df["StrategyTag"].unique()
            strategies = [s for s in available_strategies if s in valid_strategies]

            prefixes = {s[0] for s in strategies}
            if len(prefixes) != 1:
                st.error("Error: Mixed or no valid series (B, D, or G) detected in the data.")
                return
            series = prefixes.pop()
            if series == 'B':
                strategies = ['B1', 'B2', 'B3', 'B4', 'B5']
            elif series == 'D':
                strategies = ['D1', 'D2', 'D3', 'D4', 'D5']
            elif series == 'G':
                strategies = ['G1', 'G2', 'G3', 'G4', 'G5']
            else:
                st.error("Error: Invalid series detected.")
                return

            stoxo_df = stoxo_df[stoxo_df["StrategyTag"].isin(strategies)]
            user_types = [str(col) for col in stoxo_df.columns if col != "StrategyTag"]

            allocation_df, (allocation_dict, user_info, tag_allocation_rules) = allocate_strategies(allocation_file, user_types, strategies, series)
            if allocation_df is None:
                return

            output_data = []
            for user_type in user_types:
                row_data = {
                    "userId": str(user_type),
                    "ALLOCATION": allocation_df[allocation_df["userId"] == user_type]["ALLOCATION"].iloc[0] if not allocation_df[allocation_df["userId"] == user_type].empty else 0,
                    "Max Loss": int(allocation_df[allocation_df["userId"] == user_type]["Max Loss"].iloc[0]) if not allocation_df[allocation_df["userId"] == user_type].empty else 0,
                    "Tag": int(user_info.get(user_type, {}).get('tag', 0)) if user_info.get(user_type, {}).get('tag') else None,
                    "Capital Deployed": int(user_info.get(user_type, {}).get('capital', 0)),
                    "Expected Max Loss": int(user_info.get(user_type, {}).get('expected_max_loss', 0)),
                    "Allocation Valid": False,
                    "Max Loss Valid": False,
                    "Tag Valid": False,
                    "Error Message": str(user_info.get(user_type, {}).get('error', ''))
                }

                if user_type in user_info:
                    info = user_info[user_type]
                    row_data["Allocation Valid"] = info['error'] is None or "Extrapolated" in str(info['error'])
                    if info['expected_max_loss'] is not None:
                        row_data["Max Loss Valid"] = abs(info['max_loss'] - info['expected_max_loss']) < 0.01

                alloc_row = allocation_df[allocation_df["userId"] == str(user_type)]
                alloc_values = {strategy: alloc_row[strategy].iloc[0] for strategy in strategies} if not alloc_row.empty else {strategy: 0 for strategy in strategies}
                
                user_tag = user_info.get(user_type, {}).get('tag', None)
                expected_strategies = {}
                if user_tag and tag_allocation_rules.get(series) and user_tag in tag_allocation_rules[series]:
                    rule = tag_allocation_rules[series][user_tag]
                    for idx, strategy in enumerate(strategies, 1):
                        if rule[str(idx)] > 1:
                            expected_strategies[strategy] = rule[str(idx)]

                tag_valid = True
                for strategy in strategies:
                    stoxo_row = stoxo_df[stoxo_df["StrategyTag"] == strategy]
                    stoxo_calc = stoxo_row[user_type].iloc[0] if not stoxo_row.empty and user_type in stoxo_row.columns else 0
                    calculated_alloc = alloc_values[strategy]
                    values_match = (calculated_alloc == stoxo_calc)
                    row_data[f"{strategy}_Match"] = values_match
                    row_data[f"{strategy}_Calculated"] = calculated_alloc
                    row_data[f"{strategy}_Stoxo"] = stoxo_calc
                    if not values_match:
                        tag_valid = False
                
                row_data["Tag Valid"] = tag_valid
                output_data.append(row_data)

            columns = [
                "userId", "ALLOCATION", "Max Loss", "Tag", "Capital Deployed", "Expected Max Loss",
                "Allocation Valid", "Max Loss Valid", "Tag Valid", "Error Message"
            ]
            for strategy in strategies:
                columns.extend([f"{strategy}_Match", f"{strategy}_Calculated", f"{strategy}_Stoxo"])
            output_df = pd.DataFrame(output_data, columns=columns)
            output_df['userId'] = output_df['userId'].astype(str)

            st.subheader("Processed Output")
            st.dataframe(output_df)  # Display DataFrame without styling

            output_file = "processed_output.xlsx"
            output_df.to_excel(output_file, index=False)
            with open(output_file, "rb") as file:
                st.download_button(
                    label="Download Processed Output",
                    data=file,
                    file_name="processed_output.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                )

if __name__ == "__main__":
    main()
