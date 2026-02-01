from math_verify import parse, verify

def get_wrapped_content(text):
    if text and text[0] == "{":
        stack = 1
        ans = ""
        for c in text[1:]:
            if c == "{":
                stack += 1
            elif c == "}":
                stack -= 1
                if stack == 0:
                    return ans
            ans += c
    return ""

def extract_last_boxed(text):
    last = text.split("\\boxed")[-1]
    return get_wrapped_content(last)

def extract_all_boxed(text):
    splitted = text.split("\\boxed")
    found_boxeds = []
    for part in splitted[1:]:
        ans = get_wrapped_content(part)
        if ans: found_boxeds.append(ans)
    return found_boxeds if found_boxeds else [""]

def get_acc_list(predictions, gt, return_parsed=False):
    gt_parsed = parse(f"${gt}$")
    last_boxs = [extract_last_boxed(pred) for pred in predictions]
    acc_list = [verify(gt_parsed, parse(f"${pred}$")) for pred in last_boxs]
    return acc_list if not return_parsed else (acc_list, last_boxs)

def get_acc_forgive_list(predictions, gt, return_parsed=False):
    gt_parsed = parse(f"${gt}$")
    all_boxs = [extract_all_boxed(pred) for pred in predictions]
    acc_list, answers = [], []
    for boxs in all_boxs:
        is_correct = False
        for box in boxs:
            if verify(gt_parsed, parse(f"${box}$")):
                is_correct = True
                acc_list.append(True)
                answers.append(box)
                break
        if not is_correct:
            acc_list.append(False)
            answers.append(boxs[-1])
    return acc_list if not return_parsed else (acc_list, answers)

if __name__ == "__main__":
    # Example usage
    text = "The answer is \\boxed{42} and \\boxed{24}."
    print(extract_last_boxed(text))  # Output: 42
    print(extract_all_boxed(text))    # Output: ['42', '24']
    
    text = "No boxed content here."
    print(extract_last_boxed(text))  # Output: ""
    print(extract_all_boxed(text))    # Output: [""]

    gt = "068"
    predictions = [
        "Therefore, the answer is $\\boxed{68}$.",
        "The final result is \\boxed{\\frac{680}{10}}.",
        "The answer is \\boxed{68} and \\boxed{86}.",
    ]
    acc_list = get_acc_list(predictions, gt)
    print(acc_list)  # Output: [True, True, False]

    acc_list_forgive = get_acc_forgive_list(predictions, gt)
    print(acc_list_forgive)  # Output: [True, True, True]