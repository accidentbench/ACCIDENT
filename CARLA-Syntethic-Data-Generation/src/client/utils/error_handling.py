import logging
import threading
import time
import traceback
from functools import wraps

logger = logging.getLogger("runner")


class TimeoutException(Exception):
    pass


def timeout(seconds):
    """Timeout decorator using threading."""

    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Container for storing the result or exception from the thread
            results = []

            def worker():
                try:
                    results.append(func(*args, **kwargs))
                except Exception as e:
                    results.append(e)

            thread = threading.Thread(target=worker)
            thread.daemon = (
                True  # Set daemon to avoid hanging the program if it doesn't finish
            )
            thread.start()
            thread.join(seconds)

            if thread.is_alive():
                raise TimeoutException(
                    f"Function '{func.__name__}' timed out after {seconds} seconds"
                )

            result = results[0]
            if isinstance(result, Exception):
                raise result
            return result

        return wrapper

    return decorator


def retry(
    num_retries: int = 3,
    exceptions: tuple = (Exception,),
    delay: float = 5.0,
    init_message: str = None,
    retry_message: str = None,
    fail_message: str = None,
):
    """A decorator function for retrying a decorated function.

    :param num_retries: Number of attempts to call the decorated function.
    :param exceptions: Exceptions to catch while calling the decorated function.
    :param delay: Time to sleep between retires.
    :param init_message: Initial message to log.
    :param retry_message: Message to log when the decorated function raises exception.
    :param fail_message: Message to log if all attempts failed.
    :return: A decorator function.
    """
    assert num_retries > 1
    assert delay > 0

    def decorator_wrapper(func):
        if retry_message is None:
            _retry_message = f"Failed calling function {func.__name__}. Retrying..."
        else:
            _retry_message = retry_message

        if fail_message is None:
            _fail_message = f"Failed calling function {func.__name__} "
        else:
            _fail_message = fail_message

        @wraps(func)
        def decorator(*args, **kwargs):
            if init_message is not None:
                logger.info(init_message)
            error = None
            caught_exception = None
            for _ in range(num_retries):
                try:
                    return func(*args, **kwargs)
                except exceptions as e:
                    caught_exception = e
                    error = traceback.format_exc()
                    logger.info(_retry_message)
                    time.sleep(delay)
            logger.error(
                f"{_fail_message} after {num_retries} attempts. Error:\n{error}"
            )
            raise caught_exception

        return decorator

    return decorator_wrapper


def timeout_with_retry(timeout_secs, retries=3, delay=1, exceptions=(Exception,)):
    """Combines timeout and retry decorators. Timeout is first."""

    def decorator(func):
        @retry(num_retries=retries, delay=delay, exceptions=exceptions)
        @timeout(timeout_secs)
        def wrapper(*args, **kwargs):
            return func(*args, **kwargs)

        return wrapper

    return decorator
