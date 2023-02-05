class CalculOverLapping:
    # Calculate the area of the overlapping part
    # cube1 and cube2 are lists like:[xmin, xmax, ymin, ymax]
    # cube1 must be the true bndbox, cube2 is the predicted bndbox
    @staticmethod
    def calareaoverlapped(cube1, cube2):
        if not CalculOverLapping.overlappedornot(cube1, cube2):
            return 0

        xlist = [cube1[0], cube1[1], cube2[0], cube2[1]]
        ylist = [cube1[2], cube1[3], cube2[2], cube2[3]]

        xlist.sort()
        ylist.sort()

        overlapped = (xlist[2] - xlist[1]) * (ylist[2] - ylist[1])
        areacube1 = (cube1[0] - cube1[1]) * (cube1[2] - cube1[3])
        # return a percentage: overlap area compare to the first bndbox
        return overlapped / areacube1

    # See if two rectangles overlap
    @staticmethod
    def overlappedornot(cube1, cube2):
        center1 = [(cube1[0] + cube1[1]) / 2, (cube1[2] + cube1[3]) / 2]
        center2 = [(cube2[0] + cube2[1]) / 2, (cube2[2] + cube2[3]) / 2]
        centerdiff = [abs(center1[0] - center2[0]), abs(center1[1] - center2[1])]

        xsidesum = (cube1[1] - cube1[0]) / 2 + (cube2[1] - cube2[0]) / 2
        ysidesum = (cube1[3] - cube1[2]) / 2 + (cube2[3] - cube2[2]) / 2

        if (centerdiff[0] < xsidesum) & (centerdiff[1] < ysidesum):
            return True
        else:
            return False
