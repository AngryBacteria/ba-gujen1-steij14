import '@/assets/base.css'
import 'primeicons/primeicons.css'
import { createApp } from 'vue'
import { createPinia } from 'pinia'
import App from './App.vue'
import router from './router'
import PrimeVue from 'primevue/config'
import Aura from 'primevue/themes/aura'
import Button from 'primevue/button'
import Checkbox from 'primevue/checkbox'
import Menubar from 'primevue/menubar'
import Textarea from 'primevue/textarea'
import Select from 'primevue/select'
import Tooltip from 'primevue/tooltip'

const app = createApp(App)

app.use(createPinia())
app.use(router)

// PrimeVue
app.use(PrimeVue, {
  theme: {
    preset: Aura,
    options: {
      darkModeSelector: '.my-app-dark'
    }
  }
})
app.component('Button', Button)
app.component('Checkbox', Checkbox)
app.component('Menubar', Menubar)
app.component('Textarea', Textarea)
app.component('Select', Select)
app.directive('tooltip', Tooltip)

app.mount('#app')
