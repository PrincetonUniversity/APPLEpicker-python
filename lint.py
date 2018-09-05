import glob

from pylint.lint import Run


all_python_files = glob.glob('**/*.py', recursive=True)
results = Run(all_python_files, do_exit=False)

score = round(results.linter.stats['global_note'], 2)

if score < 8:
    raise Exception("Your code lint check scored only {}! "
                    "You probably want to check Lint's messages "
                    "and adjust your code.".format(score))
