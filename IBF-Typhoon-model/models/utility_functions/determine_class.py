def determine_class(x, classes):

    for key, value in classes.items():

        if value[0] <= x < value[1]:

            class_value = key
            break

    return class_value