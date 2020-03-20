import numpy as np
import pandas as pd 
import shelve
import os
import datetime


class Condor_Job:

	# Takes an iterable containing the 
	def __init__(job_name = None, func):

		# Set up a few variables to do with the job
		self.job_name = job_name
		self.function = func


	def initialise(self, whole_data):

		# Create the place to save the input and output data
		# segment the dataset into the 
		if self.job_name != None:

			self.job_name = datetime.datetime.now()


		self.file_name = os.path.join(os.cwd(), self.job_name)

		if os.path.exists(self.file_name) == False:

			os.mkdir(self.file_name)


		# Given the iterable provided go through each entry and create a file associated with that split
		for i, subset in enumerate(whole_data):

			subset_file_name = os.path.join(self.file_name, "input{}".format(i + 1))

			# Open a file and then store the data inside
			with shelve.open(subset_file_name, "n") as shelf:

				shelf["data"] = subset_

			del shelf


	def submission_file(self):

		# Generates submission file

		with open(os.path.join(os.cwd(), "run_product", "wb")) as file:

			file.write("python_script = Rubber_Band_Condor.py")
			file.write("indexed_input_files = input.dat")
			file.write("indexed_output_files = output.dat")
			file.write("log = log.txt")
			file.write("total_jobs = {}".format(n_jobs))








