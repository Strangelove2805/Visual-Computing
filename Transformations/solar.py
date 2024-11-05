import transforms as trans


def transform_earth(theta):
    """Rotates the Earth in the solar system."""
    return trans.rotation(theta)

def transform_moon1(theta):

    """Rotates moon #1 at 5 units distance from earth with tidal locking."""
    return transform_earth(theta) @ trans.translation([5, 0])


def transform_moon2(theta):
    ## TASK: Replace this code with your own implementation.
    return transform_earth(theta*2) @ trans.translation([3, 0])


def transform_moon3(theta):
    ## TASK: Replace this code with your own implementation.
    return transform_moon2(theta) @ trans.rotation(theta * 2) @ trans.translation([1, 0])
