import numpy as np

def assert_answer_1(answer):
    if answer.shape[0] == 10 and answer.shape[1] == 4:
        print("Shape is correct!")
    else:
        print("Incorrect shape. Make sure to return a 10x4 array. That is, 10 rows and 4 columns.")
    
    if np.equal(answer[0, 0], 0.417022004702574):
        print("Values are correct!")
    else:
        print("Incorrect values returned. Remember that the question asks for random numbers from a normal distribution and not a uniform distribution.\n\nHint: Check out numpy.randn() function.")


def assert_answer_2(answer):
    correct_answer = np.array([
        5.073657020217231, 6.897088786574058,
        1.9803242470482063, 4.4198267691932065])
    
    if not(len(answer.shape) == 1 and answer.shape[0] == 4):
        print("Incorrect shape. Make sure to return a 4-element array.")
    else:
        if np.all(np.equal(answer, correct_answer)):
            print("Passed!")
        else:
            print("Incorrect answer. Correct answer should be:\n{}".format(correct_answer))


def assert_answer_3(answer):
    correct_answer = np.array([
        0.9578895301505019, 0.9682615757193975,
        0.6918771139504734, 0.7501443149449675])
    
    if not(len(answer.shape) == 1 and answer.shape[0] == 4):
        print("Incorrect shape. Make sure to return a 4-element array.")
    else:
        if np.all(np.equal(answer, correct_answer)):
            print("Passed!")
        else:
            print("Incorrect answer. Correct answer should be:\n{}".format(correct_answer))



def assert_answer_4(answer):
    correct_answer = np.array([-28.33, -13.8 , -97.36,   0.  ])
    if not(len(answer.shape) == 1 and answer.shape[0] == 4):
        print("Incorrect shape. Make sure to return a 4-element array.")
    else:
        if np.all(np.equal(answer, correct_answer)):
            print("Passed!")
        else:
            print("Incorrect answer. Correct answer should be:\n{}".format(correct_answer))


def assert_answer_5(answer):
    correct_answer = np.array([[0.5267328826778572],
                               [0.28769331953003774],
                               [0.5170900904049752]])
    if not(len(answer.shape) == 2 and answer.shape[0] == 3 and answer.shape[1] == 1):
        print("Incorrect shape. Make sure to return a 3x1 array. That is, 3 rows and 1 column.")
    else:
        if np.all(np.equal(answer, correct_answer)):
            print("Passed!")
        else:
            print("Incorrect answer. Correct answer should be:\n{}".format(correct_answer))