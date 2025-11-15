from blockchain_client import contract, HOSPITAL_ACCOUNTS

count = contract.functions.getLogsCount().call()
print(f"Total blockchain logs: {count}")

h1_addr = HOSPITAL_ACCOUNTS["hospital_1"]["address"].lower()
h2_addr = HOSPITAL_ACCOUNTS["hospital_2"]["address"].lower()

h1_count = 0
h2_count = 0

print("\nContribution breakdown:")
for i in range(count):
    log = contract.functions.logs(i).call()
    addr = log[0].lower()
    if addr == h1_addr:
        h1_count += 1
        print(f"  Log {i}: hospital_1")
    elif addr == h2_addr:
        h2_count += 1
        print(f"  Log {i}: hospital_2")
    else:
        print(f"  Log {i}: unknown ({addr})")

print(f"\nSummary:")
print(f"  Hospital 1: {h1_count} contributions = {h1_count * 10} MCT")
print(f"  Hospital 2: {h2_count} contributions = {h2_count * 10} MCT")
print(
    f"  Difference: {h1_count - h2_count} contributions = {(h1_count - h2_count) * 10} MCT"
)
