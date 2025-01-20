/*
Copyright 2023.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/

package kfto

import (
	"bytes"
	"fmt"
	"testing"
	"time"

	kftov1 "github.com/kubeflow/training-operator/pkg/apis/kubeflow.org/v1"
	. "github.com/onsi/gomega"
	. "github.com/project-codeflare/codeflare-common/support"

	corev1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/api/resource"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
)

func TestPyTorchJobMnistMultiNodeSingleCpu(t *testing.T) {
	runKFTOPyTorchMnistJob(t, CPU, GetCudaTrainingImage(), "resources/requirements.txt", 2, 1)
}
func TestPyTorchJobMnistMultiNodeMultiCpu(t *testing.T) {
	runKFTOPyTorchMnistJob(t, CPU, GetCudaTrainingImage(), "resources/requirements.txt", 2, 2)
}

func TestPyTorchJobMnistMultiNodeSingleGpuWithCuda(t *testing.T) {
	runKFTOPyTorchMnistJob(t, NVIDIA, GetCudaTrainingImage(), "resources/requirements.txt", 1, 1)
}

func TestPyTorchJobMnistMultiNodeMultiGpuWithCuda(t *testing.T) {
	runKFTOPyTorchMnistJob(t, NVIDIA, GetCudaTrainingImage(), "resources/requirements.txt", 1, 2)
}

func TestPyTorchJobMnistMultiNodeSingleGpuWithROCm(t *testing.T) {
	runKFTOPyTorchMnistJob(t, AMD, GetROCmTrainingImage(), "resources/requirements-rocm.txt", 1, 1)
}

func TestPyTorchJobMnistMultiNodeMultiGpuWithROCm(t *testing.T) {
	runKFTOPyTorchMnistJob(t, AMD, GetROCmTrainingImage(), "resources/requirements-rocm.txt", 1, 2)
}

func runKFTOPyTorchMnistJob(t *testing.T, accelerator Accelerator, image string, requirementsFile string, workerReplicas, numProcPerNode int) {
	test := With(t)

	// Create a namespace
	namespace := test.NewTestNamespace()

	mnist := ReadFile(test, "resources/mnist.py")
	requirementsFileName := ReadFile(test, requirementsFile)

	if accelerator.isGpu() {
		mnist = bytes.Replace(mnist, []byte("accelerator=\"has to be specified\""), []byte("accelerator=\"gpu\""), 1)
	} else {
		mnist = bytes.Replace(mnist, []byte("accelerator=\"has to be specified\""), []byte("accelerator=\"cpu\""), 1)
	}
	config := CreateConfigMap(test, namespace.Name, map[string][]byte{
		// MNIST Ray Notebook
		"mnist.py":         mnist,
		"requirements.txt": requirementsFileName,
	})

	// Create training PyTorch job
	tuningJob := createKFTOPyTorchMnistJob(test, namespace.Name, *config, accelerator, workerReplicas, numProcPerNode, image)
	defer test.Client().Kubeflow().KubeflowV1().PyTorchJobs(namespace.Name).Delete(test.Ctx(), tuningJob.Name, *metav1.NewDeleteOptions(0))

	// Make sure the PyTorch job is running
	test.Eventually(PyTorchJob(test, namespace.Name, tuningJob.Name), TestTimeoutDouble).
		Should(WithTransform(PyTorchJobConditionRunning, Equal(corev1.ConditionTrue)))

	// Verify GPU utilization
	if IsOpenShift(test) && accelerator == NVIDIA {
		trainingPods := GetPods(test, namespace.Name, metav1.ListOptions{LabelSelector: "training.kubeflow.org/job-name=" + tuningJob.GetName()})
		test.Expect(trainingPods).To(HaveLen(workerReplicas + 1)) // +1 is a master node

		for _, trainingPod := range trainingPods {
			// Check that GPUs for training pods were utilized recently
			test.Eventually(OpenShiftPrometheusGpuUtil(test, trainingPod, accelerator), 15*time.Minute).
				Should(
					And(
						HaveLen(numProcPerNode),
						ContainElement(
							// Check that at least some GPU was utilized on more than 20%
							HaveField("Value", BeNumerically(">", 20)),
						),
					),
				)
		}
		test.T().Log("All GPUs were successfully utilized")
	}

	// Make sure the PyTorch job succeeded
	test.Eventually(PyTorchJob(test, namespace.Name, tuningJob.Name), TestTimeoutDouble).Should(WithTransform(PyTorchJobConditionSucceeded, Equal(corev1.ConditionTrue)))
	test.T().Logf("PytorchJob %s/%s ran successfully", tuningJob.Namespace, tuningJob.Name)

}

func createKFTOPyTorchMnistJob(test Test, namespace string, config corev1.ConfigMap, accelerator Accelerator, workerReplicas int, numProcPerNode int, baseImage string) *kftov1.PyTorchJob {
	var backend string
	if accelerator.isGpu() {
		backend = "nccl"
	} else {
		backend = "gloo"
	}

	tuningJob := &kftov1.PyTorchJob{
		TypeMeta: metav1.TypeMeta{
			APIVersion: corev1.SchemeGroupVersion.String(),
			Kind:       "PyTorchJob",
		},
		ObjectMeta: metav1.ObjectMeta{
			GenerateName: "kfto-mnist-",
		},
		Spec: kftov1.PyTorchJobSpec{
			PyTorchReplicaSpecs: map[kftov1.ReplicaType]*kftov1.ReplicaSpec{
				kftov1.PyTorchJobReplicaTypeMaster: {
					Replicas:      Ptr(int32(1)),
					RestartPolicy: kftov1.RestartPolicyOnFailure,
					Template: corev1.PodTemplateSpec{
						ObjectMeta: metav1.ObjectMeta{
							Labels: map[string]string{
								"app":  "kfto-mnist",
								"role": "master",
							},
						},
						Spec: corev1.PodSpec{
							Affinity: &corev1.Affinity{
								PodAntiAffinity: &corev1.PodAntiAffinity{
									RequiredDuringSchedulingIgnoredDuringExecution: []corev1.PodAffinityTerm{
										{
											LabelSelector: &metav1.LabelSelector{
												MatchLabels: map[string]string{
													"app": "kfto-mnist",
												},
											},
											TopologyKey: "kubernetes.io/hostname",
										},
									},
								},
							},
							Containers: []corev1.Container{
								{
									Name:            "pytorch",
									Image:           baseImage,
									ImagePullPolicy: corev1.PullIfNotPresent,
									Command: []string{
										"/bin/bash", "-c",
										fmt.Sprintf(`mkdir -p /tmp/lib /tmp/datasets/mnist && export PYTHONPATH=$PYTHONPATH:/tmp/lib && \
										pip install --no-cache-dir -r /mnt/files/requirements.txt --target=/tmp/lib && \
										echo "Downloading MNIST dataset..." && \
										python3 -c "from torchvision.datasets import MNIST; from torchvision.transforms import Compose, ToTensor; \
										MNIST('/tmp/datasets/mnist', train=False, download=True, transform=Compose([ToTensor()]))" && \
										echo -e "\n\n Dataset downloaded to /tmp/datasets/mnist" && ls -R /tmp/datasets/mnist && \
										echo -e "\n\n Starting training..." && \
										torchrun --nproc_per_node=%d /mnt/files/mnist.py --dataset_path "/tmp/datasets/mnist" --epochs 7 --save_every 2 --batch_size 128 --lr 0.001 --snapshot_path "mnist_snapshot.pt" --backend %s`, numProcPerNode, backend),
									},
									VolumeMounts: []corev1.VolumeMount{
										{
											Name:      config.Name,
											MountPath: "/mnt/files",
										},
										{
											Name:      "tmp-volume",
											MountPath: "/tmp",
										},
									},
									Resources: corev1.ResourceRequirements{
										Requests: corev1.ResourceList{
											corev1.ResourceCPU:    resource.MustParse(fmt.Sprintf("%d", numProcPerNode)),
											corev1.ResourceMemory: resource.MustParse("6Gi"),
										},
										Limits: corev1.ResourceList{
											corev1.ResourceCPU:    resource.MustParse(fmt.Sprintf("%d", numProcPerNode)),
											corev1.ResourceMemory: resource.MustParse("6Gi"),
										},
									},
								},
							},
							Volumes: []corev1.Volume{
								{
									Name: config.Name,
									VolumeSource: corev1.VolumeSource{
										ConfigMap: &corev1.ConfigMapVolumeSource{
											LocalObjectReference: corev1.LocalObjectReference{
												Name: config.Name,
											},
										},
									},
								},
								{
									Name: "tmp-volume",
									VolumeSource: corev1.VolumeSource{
										EmptyDir: &corev1.EmptyDirVolumeSource{},
									},
								},
							},
							RestartPolicy: corev1.RestartPolicyOnFailure,
						},
					},
				},
				kftov1.PyTorchJobReplicaTypeWorker: {
					Replicas:      Ptr(int32(workerReplicas)),
					RestartPolicy: kftov1.RestartPolicyOnFailure,
					Template: corev1.PodTemplateSpec{
						ObjectMeta: metav1.ObjectMeta{
							Labels: map[string]string{
								"app":  "kfto-mnist",
								"role": "worker",
							},
						},
						Spec: corev1.PodSpec{
							Affinity: &corev1.Affinity{
								PodAntiAffinity: &corev1.PodAntiAffinity{
									RequiredDuringSchedulingIgnoredDuringExecution: []corev1.PodAffinityTerm{
										{
											LabelSelector: &metav1.LabelSelector{
												MatchLabels: map[string]string{
													"app": "kfto-mnist",
												},
											},
											TopologyKey: "kubernetes.io/hostname",
										},
									},
								},
							},
							Containers: []corev1.Container{
								{
									Name:            "pytorch",
									Image:           baseImage,
									ImagePullPolicy: corev1.PullIfNotPresent,
									Command: []string{
										"/bin/bash", "-c",
										fmt.Sprintf(`mkdir -p /tmp/lib /tmp/datasets/mnist && export PYTHONPATH=$PYTHONPATH:/tmp/lib && \
										pip install --no-cache-dir -r /mnt/files/requirements.txt --target=/tmp/lib && \
										echo "Downloading MNIST dataset..." && \
										python3 -c "from torchvision.datasets import MNIST; from torchvision.transforms import Compose, ToTensor; \
										MNIST('/tmp/datasets/mnist', train=False, download=True, transform=Compose([ToTensor()]))" && \
										echo -e "\n\n Dataset downloaded to /tmp/datasets/mnist" && ls -R /tmp/datasets/mnist && \
										echo -e "\n\n Starting training..." && \
										torchrun --nproc_per_node=%d /mnt/files/mnist.py --dataset_path "/tmp/datasets/mnist" --epochs 7 --save_every 2 --batch_size 128 --lr 0.001 --snapshot_path "mnist_snapshot.pt" --backend %s`, numProcPerNode, backend),
									},
									VolumeMounts: []corev1.VolumeMount{
										{
											Name:      config.Name,
											MountPath: "/mnt/files",
										},
										{
											Name:      "tmp-volume",
											MountPath: "/tmp",
										},
									},
									Resources: corev1.ResourceRequirements{
										Requests: corev1.ResourceList{
											corev1.ResourceCPU:    resource.MustParse(fmt.Sprintf("%d", numProcPerNode)),
											corev1.ResourceMemory: resource.MustParse("6Gi"),
										},
										Limits: corev1.ResourceList{
											corev1.ResourceCPU:    resource.MustParse(fmt.Sprintf("%d", numProcPerNode)),
											corev1.ResourceMemory: resource.MustParse("6Gi"),
										},
									},
								},
							},
							Volumes: []corev1.Volume{
								{
									Name: config.Name,
									VolumeSource: corev1.VolumeSource{
										ConfigMap: &corev1.ConfigMapVolumeSource{
											LocalObjectReference: corev1.LocalObjectReference{
												Name: config.Name,
											},
										},
									},
								},
								{
									Name: "tmp-volume",
									VolumeSource: corev1.VolumeSource{
										EmptyDir: &corev1.EmptyDirVolumeSource{},
									},
								},
							},
							RestartPolicy: corev1.RestartPolicyOnFailure,
						},
					},
				},
			},
		},
	}

	if accelerator.isGpu() {
		// Update resource lists for GPU (NVIDIA/ROCm) usecase
		tuningJob.Spec.PyTorchReplicaSpecs["Master"].Template.Spec.Containers[0].Resources.Requests[corev1.ResourceName(accelerator.ResourceLabel)] = resource.MustParse(fmt.Sprint(numProcPerNode))
		tuningJob.Spec.PyTorchReplicaSpecs["Master"].Template.Spec.Containers[0].Resources.Limits[corev1.ResourceName(accelerator.ResourceLabel)] = resource.MustParse(fmt.Sprint(numProcPerNode))
		tuningJob.Spec.PyTorchReplicaSpecs["Worker"].Template.Spec.Containers[0].Resources.Requests[corev1.ResourceName(accelerator.ResourceLabel)] = resource.MustParse(fmt.Sprint(numProcPerNode))
		tuningJob.Spec.PyTorchReplicaSpecs["Worker"].Template.Spec.Containers[0].Resources.Limits[corev1.ResourceName(accelerator.ResourceLabel)] = resource.MustParse(fmt.Sprint(numProcPerNode))

		tuningJob.Spec.PyTorchReplicaSpecs["Master"].Template.Spec.Containers[0].Env = []corev1.EnvVar{
			{
				Name:  "NCCL_DEBUG",
				Value: "INFO",
			},
			{
				Name:  "TORCH_DISTRIBUTED_DEBUG",
				Value: "DETAIL",
			},
		}
		tuningJob.Spec.PyTorchReplicaSpecs["Worker"].Template.Spec.Containers[0].Env = []corev1.EnvVar{
			{
				Name:  "NCCL_DEBUG",
				Value: "INFO",
			},
			{
				Name:  "TORCH_DISTRIBUTED_DEBUG",
				Value: "DETAIL",
			},
		}

		// Update tolerations
		tuningJob.Spec.PyTorchReplicaSpecs["Master"].Template.Spec.Tolerations = []corev1.Toleration{
			{
				Key:      accelerator.ResourceLabel,
				Operator: corev1.TolerationOpExists,
			},
		}
		tuningJob.Spec.PyTorchReplicaSpecs["Worker"].Template.Spec.Tolerations = []corev1.Toleration{
			{
				Key:      accelerator.ResourceLabel,
				Operator: corev1.TolerationOpExists,
			},
		}
	}

	tuningJob, err := test.Client().Kubeflow().KubeflowV1().PyTorchJobs(namespace).Create(test.Ctx(), tuningJob, metav1.CreateOptions{})
	test.Expect(err).NotTo(HaveOccurred())
	test.T().Logf("Created PytorchJob %s/%s successfully", tuningJob.Namespace, tuningJob.Name)

	return tuningJob
}
